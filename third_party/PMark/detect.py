import json
import os
import tempfile
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
import torch
from tqdm import tqdm
import argparse

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from vllm import LLM

from utils.detect import detect_paragraph
from utils.rand import secret_mbit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="logs/opt-1.3b_log")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_context", type=int, default=1024)
    parser.add_argument("--detect_origin", type=bool, default=False, help="whether including result of natural text")
    parser.add_argument("--sub_dir", type=str, default=None)
    parser.add_argument("--pivot", type=str, default="rand")
    parser.add_argument("--dedup", type=bool, default=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--median_method", type=str, default="torch")
    parser.add_argument("--embedder_name", type=str, default=None)
    parser.add_argument("--msig", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()
    # save args
    os.makedirs(f"{args.log_dir}/{args.sub_dir}/detect", exist_ok=True)
    if not os.path.exists(f"{args.log_dir}/{args.sub_dir}/detect/config.json"):
        with open(f"{args.log_dir}/{args.sub_dir}/detect/config.json", "w", encoding="utf-8") as f:
            json.dump(args.__dict__, f, ensure_ascii=False, indent=4)
    _max_len = os.environ.get("PMARK_MAX_MODEL_LEN")
    llm = LLM(model=args.model_name, gpu_memory_utilization=0.7,
              tensor_parallel_size=1,
              **({"max_model_len": int(_max_len)} if _max_len else {}))
    # embedder
    if "sonar" in args.embedder_name:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder", 
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
    else:
        embedder=SentenceTransformer(args.embedder_name, device="cuda")
    
    # tokenizer
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.msig>1:
        secret_mbit.set_signum(args.msig)
    if args.median_method=="hd":
        args.dedup=False

    log_dir=args.log_dir
    sub_dir=args.sub_dir
    origin_pos=[]
    watermark_pos=[]
    for i in tqdm(range(args.start, args.end)):
        if not os.path.exists(f"{log_dir}/{sub_dir}/{i}.json"):
            print(f"{log_dir}/{sub_dir}/{i}.json")
            print(f"Sample {i} not found!")
            continue
        print(f"Processing sample {i}")
        detection_path = f"{log_dir}/{sub_dir}/detect/{i}.json"
        if os.path.exists(detection_path):
            # Detection is document-local. Reusing a completed JSON preserves
            # the exact official score while making crash recovery incremental.
            continue
        with open(f"{log_dir}/{sub_dir}/{i}.json", "r", encoding="utf-8") as f:
            generated_log = json.load(f)
        gen_sentences=[d['text'] for d in generated_log['log']]
        gen_sentences.insert(0, generated_log['prompt'])

        watermarked_text=generated_log['prompt']+generated_log['generated_text']
        detect_sentences=sent_tokenize(watermarked_text)

        if len(detect_sentences) != len(gen_sentences): # segmentation inconsistent
            print(f"WARNING: Sample {i}, generated sentecnes {len(gen_sentences)}, detected sentecnes {len(detect_sentences)}!!!")

        watermark_log = detect_paragraph(gen_sentences, num_samples=args.num_samples, llm=llm, embedder=embedder, pivot=args.pivot, median_method=args.median_method, debug=True, dedup=args.dedup, msig=args.msig, temperature=args.temperature, top_p=args.top_p)
        is_watermarked2=watermark_log["is_watermarked"]
        watermark_pos.append(is_watermarked2)
        print(f"Watermarked text detection finished!")
        print(f"Watermarked text is watermarked: {is_watermarked2}")

        origin_log=None
        if args.detect_origin:
            original_text=generated_log["original_text"]
            origin_sens=sent_tokenize(original_text)
            
            origin_ids=tokenizer(original_text, return_tensors="pt")["input_ids"]
            if origin_ids.shape[1]>args.max_context:
                print("WARNING: original text is too long, truncating...")
                origin_ids=origin_ids[:,:args.max_context-500]
            original_text=tokenizer.decode(origin_ids[0])

            if len(origin_sens)>12:
                origin_sens=origin_sens[:12]

            origin_log = detect_paragraph(origin_sens, num_samples=args.num_samples, model_name=args.model_name, llm=llm, embedder=embedder, pivot=args.pivot, median_method=args.median_method, debug=True, dedup=args.dedup, msig=args.msig, temperature=args.temperature, top_p=args.top_p)
            is_watermarked1=origin_log["is_watermarked"]
            origin_pos.append(is_watermarked1)
            print(f"Origin text detection finished!")
            print(f"Origin text is watermarked: {is_watermarked1}\n")
        print("******************")
        res_log={"origin_log":origin_log,"watermark_log":watermark_log}
        os.makedirs(f"{log_dir}/{sub_dir}/detect", exist_ok=True)
        detection_dir = f"{log_dir}/{sub_dir}/detect"
        os.makedirs(detection_dir, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{i}.", suffix=".json.tmp", dir=detection_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(res_log, f, ensure_ascii=False, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, detection_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    print("----------------")
    print("Finished!!!")
