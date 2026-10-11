'''partially adapted from SemStamp'''
# By the courtesy of the authors of 
# "SemStamp: A Semantic Watermark with Paraphrastic Robustness for Text Generation "
# https://arxiv.org/abs/2310.03991
# "k-SemStamp: A Clustering-Based Semantic Watermark for Detection of Machine-Generated Text"
# https://arxiv.org/abs/2402.11399
# original github link: https://github.com/bohanhou14/semstamp

import pprint
import argparse
import json
import os
import sys
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from semfield.liger_inference import apply_liger_to_model
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
from sampling_utils import cosine_reject_completion, euclidean_reject_completion
import pickle 
import sampling_utils
from sampling_utils import DummyPCA 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='path to huggingface dataset that has a column "text"')
    parser.add_argument('--output', type=str, help='output folder name.')
    parser.add_argument('--model', type=str, help='model name to generate continuation. huggingface/openai', default="AbeHou/opt-1.3b-semstamp")
    parser.add_argument('--embedder', type=str, help='model name to embed sentences', default='hkunlp/instructor-large')
    parser.add_argument('--max_new_tokens', type=int, default=205)
    parser.add_argument('--min_new_tokens', type=int, default=195)
    parser.add_argument('--max_trials', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--rep_p', type=float, default=1.05)
    parser.add_argument('--a', type=float, default=0.5)
    parser.add_argument('--b', type=float, default=1)
    parser.add_argument('--mode', type=str, choices=['cosine', 'euclidean'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_pca', action='store_true', default=False, help='When set, use PCA for dimensionality reduction')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--full_prompt', action='store_true',
                        help='Use the complete input string (for rendered chat prompts)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the per-document checkpoint in --output')
    pp = pprint.PrettyPrinter(indent=4)
    args = parser.parse_args()
    pp.pprint(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    is_offline = os.environ.get('TRANSFORMERS_OFFLINE') is not None and os.environ.get('TRANSFORMERS_OFFLINE') == '1'

    sampling_utils.MAX_TRIALS = args.max_trials
    dataset = load_from_disk(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=is_offline)
    folder_name = os.path.join(args.data, args.embedder)
    
    bad_words_ids = tokenizer("\n", return_tensors="pt", add_special_tokens=False).input_ids.to(device=args.device).tolist() # block \n
    
    gen_config = GenerationConfig.from_pretrained(
        args.model,
        return_dict_in_generate=True,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        repetition_penalty=args.rep_p,
        bad_words_ids=bad_words_ids,
        local_files_only=is_offline
    )

    name = args.output
    if 'cosine' in args.mode:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=is_offline).to(args.device)
        apply_liger_to_model(model)
        model.eval()
        embedder = SentenceTransformer(args.embedder)
        if args.use_pca:
            with open('pca_model_16.pkl', 'rb') as f:
                pca_model = pickle.load(f)
        else:
            pca_model = DummyPCA()
        def text_to_generated_text(ex):
            prompt = ex['text'] if args.full_prompt else sent_tokenize(ex['text'])[0]
            response= cosine_reject_completion(
                prompt=prompt,
                model=model, tokenizer=tokenizer, gen_config=gen_config,
                embedder=embedder,
                whole_interval=(args.a, args.b),
                pca_model=pca_model,
                verbose=args.verbose,
                device=args.device)
            ex['text'] = response.strip()
            return ex
    elif 'euclidean' in args.mode:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=is_offline).to(args.device)
        apply_liger_to_model(model)
        model.eval()
        embedder = SentenceTransformer(args.embedder)
        if args.use_pca:
            with open('pca_model_16.pkl', 'rb') as f:
                pca_model = pickle.load(f)
        else:
            pca_model = DummyPCA()
        def text_to_generated_text(ex):
            prompt = ex['text'] if args.full_prompt else sent_tokenize(ex['text'])[0]
            response= euclidean_reject_completion(
                prompt=prompt,
                model=model, tokenizer=tokenizer, gen_config=gen_config,
                embedder=embedder,
                whole_interval=(args.a, args.b),
                pca_model=pca_model,
                verbose=args.verbose,
                device=args.device)
            ex['text'] = response.strip()
            return ex
    else:
        raise NotImplementedError
   
    os.makedirs(name, exist_ok=True)
    checkpoint = Path(name) / "results.partial.jsonl"
    completed = {}
    if args.resume and checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row["index"])
                if index < 0 or index >= len(dataset):
                    raise ValueError(f"checkpoint index out of range at line {line_number}")
                completed[index] = str(row["text"])
        print(f"resuming {len(completed)}/{len(dataset)} completed documents", flush=True)

    new_texts = [None] * len(dataset)
    for index, example in enumerate(dataset):
        if index in completed:
            new_texts[index] = completed[index]
            continue
        result = text_to_generated_text(dict(example))
        new_texts[index] = str(result["text"]).strip()
        with checkpoint.open("a", encoding="utf-8") as handle:
            json.dump({"index": index, "text": new_texts[index]},
                      handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"SimMark document {index + 1}/{len(dataset)} checkpointed", flush=True)

    print("Saving results to", name, flush=True)
    num_sents = np.sum([len(sent_tokenize(t)) for t in new_texts])
    new_dataset = Dataset.from_dict({'text': new_texts})
    new_dataset.save_to_disk(name)
    checkpoint.unlink(missing_ok=True)
