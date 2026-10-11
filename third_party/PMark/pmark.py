# from datasets import load_dataset, load_from_disk
from vllm import LLM
from utils.sample import speculative_sample_next_sentence, sample_next_sentence_msignal, secret_mbit
import torch
import json
import os
import tempfile
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from datasets import load_from_disk
import argparse
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--log_dir", type=str, default="logs/watermark_log")
    parser.add_argument("--data_name", type=str, default="c4")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--pivot", type=str, default="rand")
    parser.add_argument("--median_method", type=str, default="torch", help="median estimation method, select from [\"torch\",\"hd\",\"kde\",\"prior\"]")
    parser.add_argument("--dedup", type=bool, default=False, help="whether to deduplicate during sampling")
    parser.add_argument("--start", type=int, default=0, help="begin of sample idx")
    parser.add_argument("--end", type=int, default=500, help="end of sample idx")
    parser.add_argument("--embedder_path", type=str, default=None, help="text encoder")
    parser.add_argument("--fast_llm", type=str, default=None, help="fast LLM used for sampling-based median estimation")
    parser.add_argument("--msig", type=int, default=0, help="channel number")
    parser.add_argument("--parallel", type=bool, default=False, help="sampling in parallel or sequentially")
    parser.add_argument("--device0", type=str, default="0", help="device for LLM")
    parser.add_argument("--device1", type=str, default="1", help="device for fast LLM(used for speculative sampling)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="optional load_from_disk dataset with prompt/text fields")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()
    # PMark uses one vLLM engine for all candidate generation.
    fast_llm=None
    # OPT-1.3B caps at 2048 positions; Llama-3.2 advertises 131072, which
    # would let vLLM size its cache and its prompts for a context this
    # benchmark never uses. PMARK_MAX_MODEL_LEN pins it so every generator
    # sees the same window. Unset keeps vLLM's own default.
    _max_len = os.environ.get("PMARK_MAX_MODEL_LEN")
    llm = LLM(model=args.model_path, gpu_memory_utilization=0.8,
              tensor_parallel_size=1,
              **({"max_model_len": int(_max_len)} if _max_len else {}))
    
    #save args
    os.makedirs(f"{args.log_dir}", exist_ok=True)
    with open(f"{args.log_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump(args.__dict__, f, ensure_ascii=False, indent=4)

    #tokenizer and embedder
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "sonar" in args.embedder_path:
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", 
            tokenizer="text_sonar_basic_encoder", 
            device=torch.device("cuda"),
            dtype=torch.float16,
        )
    else:
        embedder=SentenceTransformer(args.embedder_path, device="cuda")

    if args.fast_llm:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device1
        fast_llm = LLM(model=args.fast_llm, gpu_memory_utilization=0.8, tensor_parallel_size=1)
    
    # init random seeds
    if args.msig>1:
        secret_mbit.set_signum(args.msig)
    if args.median_method=="hd":
        args.dedup=False

    if args.dataset_path:
         dataset = load_from_disk(args.dataset_path)
    elif args.data_name=="c4":
         dataset = load_from_disk("./data/c4-val-500")
    elif args.data_name=="booksum":
            dataset = load_from_disk("./data/booksum-train-500")
    min_new_tokens=205
    min_new_sentences=12

    for i in tqdm(range(args.start, args.end)):
        if os.path.exists(f"{args.log_dir}/{i}.json"):
            continue
        text=dataset[i]["text"]
        prompt=str(dataset[i].get("prompt", "")).strip() or sent_tokenize(text)[0]

        generated_text=""
        generated_log={"prompt":prompt,"generated_text":"","original_text":text,"log":[]}
        it=0

        while True:
            it+=1
            # generate watermarked sentence
            for _ in range(5):
                if args.fast_llm:# use speculative sampling
                    next_sample=speculative_sample_next_sentence(embedder=embedder, prompt=prompt, num_samples=args.num_samples, debug=True, savefig=None, pivot=args.pivot, sen_id=it, median_method=args.median_method, llm=llm, fast_llm=fast_llm) #only support vllm implementation
                else:# multiple channel sampling
                    next_sample=sample_next_sentence_msignal(embedder=embedder, prompt=prompt, model_name=args.model_path, num_samples=args.num_samples, debug=True, sen_id=it, median_method=args.median_method, llm=llm, dedup=args.dedup, temperature=args.temperature, top_p=args.top_p)
                if next_sample:
                    break
            if not next_sample:
                break
            next_sentence=next_sample["text"]
            generated_log['log'].append(next_sample)
            # update prompt
            prompt=prompt+" "+next_sentence
            generated_text=generated_text+" "+next_sentence
            generated_log['generated_text']=generated_text

            generated_ids=tokenizer(generated_text, return_tensors="pt")["input_ids"]
            if generated_ids.shape[1]>min_new_tokens and it>=min_new_sentences:
                break

        print(f"Complete paragraph: {prompt}")
        print("******************")
        os.makedirs(f"{args.log_dir}", exist_ok=True)
        output_path = f"{args.log_dir}/{i}.json"
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{i}.", suffix=".json.tmp", dir=args.log_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(generated_log, f, ensure_ascii=False, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, output_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
