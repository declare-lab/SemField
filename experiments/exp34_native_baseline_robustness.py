"""Apply shared sentence attacks and rerun one baseline's native detector."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise


def seed(parts):
    h = hashlib.blake2b(digest_size=8)
    for part in parts: h.update(str(part).encode()); h.update(b"\0")
    return int.from_bytes(h.digest(), "big")


def rows(path):
    data = json.loads(Path(path).read_text())
    return {int(x["document_id"]): x for x in data["documents"]}


def external(path):
    if not path: return {}
    return {int(x["document_id"]): x["sentences"]
            for x in json.loads(Path(path).read_text())["documents"]}


def attack(clean, donor, kind, strength, rng, paraphrase=None):
    clean=list(clean); n=len(clean)
    if kind == "none" or strength == 0 or n == 0: return clean
    if kind == "reorder":
        moved=min(n,max(2,round(strength*n)))
        if n < 2: return clean
        chosen=np.sort(rng.choice(n,moved,replace=False)); shift=int(rng.integers(1,moved))
        source=np.arange(n); source[chosen]=np.roll(chosen,shift)
        return [clean[int(i)] for i in source]
    if kind in ("delete","crop"):
        removed=min(n,max(1,round(strength*n)))
        if kind == "delete":
            keep=np.ones(n,dtype=bool); keep[rng.choice(n,removed,replace=False)]=False
            return [x for x,k in zip(clean,keep) if k]
        retained=n-removed
        if retained <= 0: return []
        start=int(rng.integers(0,n-retained+1)); return clean[start:start+retained]
    if kind == "substitute":
        if not donor: return clean
        count=min(n,max(1,round(strength*n))); chosen=rng.choice(n,count,replace=False)
        out=clean[:]
        for i,x in zip(chosen,rng.choice(donor,count,replace=True)): out[int(i)]=str(x)
        return out
    if kind in ("dilute", "dilute-interleave"):
        if not donor: return clean
        count=max(1,n if strength >= 1 else round(strength*n/(1-strength)))
        additions=[str(x) for x in rng.choice(donor,count,replace=True)]
        if kind == "dilute-interleave":
            # Uniform random merge that preserves the order of both streams.
            # This is a diagnostic variant of the block-dilution attack: it
            # removes the single random cut used by the original attack while
            # retaining the same donor pool and insertion count.
            total = n + len(additions)
            clean_slots = set(int(i) for i in rng.choice(total, n, replace=False))
            merged=[]; clean_i=0; donor_i=0
            for slot in range(total):
                if slot in clean_slots:
                    merged.append(clean[clean_i]); clean_i += 1
                else:
                    merged.append(additions[donor_i]); donor_i += 1
            return merged
        if strength >= 1: return additions
        cut=int(rng.integers(0,count+1)); return additions[:cut]+clean+additions[cut:]
    if kind == "paraphrase":
        para=list(paraphrase or [])
        if strength >= 1: return para
        out=clean[:]; common=min(n,len(para)); count=min(common,max(1,round(strength*common)))
        for i in rng.choice(common,count,replace=False): out[int(i)]=para[int(i)]
        return out
    raise ValueError(kind)

def alimark_probing_attack(clean, kind, strength, rng):
    """AliMark probing_insert/delete/reorder, with NumPy RNG for reproducibility."""
    clean=list(clean); n=len(clean)
    if kind == 'none' or strength == 0 or n == 0: return clean
    count=min(int(strength*n),max(0,n-1))
    if kind == 'insert':
        positions=set(map(int,rng.choice(np.arange(1,n),count,replace=False))) if count else set(); out=[]
        for i,sentence in enumerate(clean):
            out.append(sentence)
            if i in positions: out.append(('.', '?', '!')[int(rng.integers(0,3))])
        return out
    positions=sorted(map(int,rng.choice(n,count,replace=False))) if count else []
    if kind == 'delete': return [x for i,x in enumerate(clean) if i not in set(positions)]
    if kind == 'reorder':
        values=[clean[i] for i in positions]; rng.shuffle(values); out=clean[:]
        for i,value in zip(positions,values): out[i]=value
        return out
    raise ValueError(kind)


class Detector:
    def __init__(self, args, clean_null):
        self.args=args; self.cwd=Path.cwd(); self.threshold=None; self.native=True
        if args.method == "simmark": self._simmark(clean_null)
        elif args.method.startswith("pmark-"): self._pmark()
        elif args.method == "alimark": self._alimark()
        elif args.method in ("kgw", "sir", "semstamp", "k-semstamp",
                             "exp", "synthid"):
            self._markllm()

    def _simmark(self, clean_null):
        base=ROOT/'third_party/SimMark'; sys.path.insert(0,str(base))
        spec=importlib.util.spec_from_file_location('simmark_detection',base/'detection.py')
        self.mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.mod)
        from sentence_transformers import SentenceTransformer
        self.encoder=SentenceTransformer('hkunlp/instructor-large',device=self.args.device)
        self.instruction='Represent the sentence for cosine similarity:'
        emb=self.mod.calculate_embeddings(clean_null,self.encoder,self.instruction)
        values=[]
        for text,e in zip(clean_null,emb):
            values.extend(self.mod.calc_dist(__import__('nltk').tokenize.sent_tokenize(text),'cosine',e,self.mod.DummyPCA()))
        hist,bins=np.histogram(np.asarray(values),bins=1000); centers=(bins[:-1]+bins[1:])/2
        self.gamma=float(hist[(centers>=.68)&(centers<=.76)].sum()/max(1,len(values)))
        self.threshold=2.33

    def _pmark(self):
        base=ROOT/'third_party/PMark'; os.chdir(base); sys.path.insert(0,str(base))
        from sentence_transformers import SentenceTransformer
        from utils.detect import detect_paragraph
        from utils.rand import secret_mbit
        self.detect_paragraph=detect_paragraph; secret_mbit.set_signum(4)
        from semfield.generation import VLLMCandidateGenerator
        self.llm=VLLMCandidateGenerator(
            self.args.model, dtype='float16',
            max_model_len=self.args.vllm_max_model_len,
            python=self.args.vllm_python,
            max_num_seqs=self.args.vllm_max_num_seqs,
            request_size=1,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            tensor_parallel_size=1)
        self.encoder=SentenceTransformer(self.args.embedder,device=self.args.device)
        self.median='hd' if self.args.method.endswith('online') else 'prior'

    def _alimark(self):
        base=ROOT/'third_party/AliMark'; os.chdir(base); sys.path.insert(0,str(base))
        from types import SimpleNamespace
        from watermark.alimark import AliMark
        cfg=SimpleNamespace(watermark_model=self.args.model,watermark_embedder=self.args.embedder,
            watermark_num_next_sentence_candidates=64,min_new_sentences=12,device=self.args.device,
            watermark_embedding_dim=int(getattr(self.args, "alimark_embedding_dim", 768)),
            watermark_block_size=8,seed=42,watermark_rs_dropout=0.0)
        self.alimark=AliMark(cfg,load_llm=False); self.native=False

    def _markllm(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from semfield.liger_inference import apply_liger_to_model
        base=ROOT/'third_party/MarkLLM'; os.chdir(base); sys.path.insert(0,str(base))
        from watermark.auto_watermark import AutoWatermark
        from utils.transformers_config import TransformersConfig
        tok=AutoTokenizer.from_pretrained(self.args.model); tok.pad_token=tok.pad_token or tok.eos_token
        from semfield.generation import default_precision_for_model
        requested_precision = getattr(self.args, "precision", None)
        if requested_precision is None:
            # Keep OPT-compatible fp16 by default while using the numerically
            # safe dtype for Llama checkpoints when this evaluator is invoked
            # directly (the generation runners pass precision explicitly).
            requested_precision = default_precision_for_model(self.args.model)
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                 "fp32": torch.float32}[requested_precision]
        model=AutoModelForCausalLM.from_pretrained(self.args.model,dtype=dtype).to(self.args.device).eval(); apply_liger_to_model(model)
        cfg=TransformersConfig(model=model,tokenizer=tok,device=self.args.device,max_new_tokens=205,
            min_new_tokens=195,do_sample=True,temperature=.7,top_p=.95,repetition_penalty=1.05,pad_token_id=tok.pad_token_id)
        name={"semstamp":"SEMSTAMP", "k-semstamp":"KSEMSTAMP",
              "synthid":"SynthID", "exp":"EXP"}.get(
                  self.args.method, self.args.method.upper())
        algorithm_config = base / "config" / f"{name}.json"
        if name == "SIR":
            # MarkLLM's default SIR.json is tied to OPT's 50,272-token
            # vocabulary.  Llama token ids can reach 128,255, so using that
            # file during detection causes an out-of-bounds lookup.  Select
            # the official vocabulary-sized SIR config/mapping instead.
            vocab_size = len(tok)
            sized_config = base / "config" / f"SIR-{vocab_size}.json"
            if sized_config.exists():
                algorithm_config = sized_config
            else:
                import json as _json
                sir_config = _json.loads(algorithm_config.read_text())
                sir_config["mapping_name"] = (
                    f"watermark/sir/mapping/300_mapping_{vocab_size}.json"
                )
                sized_config.write_text(_json.dumps(sir_config, indent=2))
                algorithm_config = sized_config
        self.mark=AutoWatermark.load(name,algorithm_config=str(algorithm_config),transformers_config=cfg)

    @torch.inference_mode()
    def score_many(self, texts, cache_key=None, prompts=None):
        if self.args.method == 'simmark':
            emb=self.mod.calculate_embeddings(texts,self.encoder,self.instruction); out=[]
            for text,e in zip(texts,emb): out.append(float(self.mod.detect(__import__('nltk').tokenize.sent_tokenize(text),'cosine',e,self.mod.DummyPCA(),self.gamma,whole_interval=(.68,.76),K=250)))
            return out,[x>self.threshold for x in out]
        if self.args.method.startswith('pmark-'):
            from nltk.tokenize import sent_tokenize
            prompts = list(prompts or [""] * len(texts))
            if len(prompts) != len(texts):
                raise ValueError("PMark prompt/text count mismatch")
            scores=[]; decisions=[]; journal=None
            contract={
                'method':self.args.method,
                'backend':'vllm',
                'model':self.args.model,
                'embedder':self.args.embedder,
                'num_samples':64,
                'texts_sha256':hashlib.sha256(
                    json.dumps(texts,ensure_ascii=False,separators=(',',':')).encode()
                ).hexdigest(),
                'prompts_sha256':hashlib.sha256(
                    json.dumps(prompts,ensure_ascii=False,separators=(',',':')).encode()
                ).hexdigest(),
                'prompt_restored':True,
            }
            if cache_key:
                safe=''.join(c if c.isalnum() or c in '.-_' else '-'
                             for c in cache_key)
                journal=Path(f'{self.args.out}.{safe}.journal.json')
                if journal.exists():
                    saved=json.loads(journal.read_text())
                    if saved.get('contract') != contract:
                        raise RuntimeError(f'PMark journal contract differs: {journal}')
                    scores=[float(x) for x in saved.get('scores',[])]
                    decisions=[bool(x) for x in saved.get('decisions',[])]
                    if len(scores) != len(decisions) or len(scores)>len(texts):
                        raise RuntimeError(f'invalid PMark journal: {journal}')
            from tqdm.auto import tqdm
            progress=tqdm(total=len(texts), initial=len(scores), desc=f"{self.args.method}",
                          unit="doc", dynamic_ncols=True)
            for index,text in enumerate(texts[len(scores):],start=len(scores)):
                random.seed(seed((self.args.seed,cache_key,index,'pmark-vllm')))
                prompt=str(prompts[index]).strip()
                sequence=([prompt] if prompt else [])+sent_tokenize(text)
                result=self.detect_paragraph(sequence,num_samples=64,embedder=self.encoder,
                    pivot='rand',median_method=self.median,dedup=(self.median!='hd'),msig=4,
                    llm=self.llm)
                scores.append(float(result['z_score'])); decisions.append(bool(result['is_watermarked']))
                if journal is not None:
                    atomic_json(journal,{'contract':contract,'scores':scores,
                                         'decisions':decisions})
                progress.update(1)
            progress.close()
            return scores,decisions
        if self.args.method == 'alimark':
            scores=[float(self.alimark.detect_watermark(x)['score']) if x.strip() else -1e9 for x in texts]
            return scores,None
        results=[]
        for text in texts:
            try:
                results.append(self.mark.detect_watermark(text))
            except ValueError as exc:
                if "at least 1 token" not in str(exc):
                    raise
                results.append({"score": -1e9, "is_watermarked": False})
        scores=[]; decisions=[]
        for result in results:
            value=float(result['score'])
            if not math.isfinite(value):
                value=-1e9; decision=False
            else:
                decision=bool(result['is_watermarked'])
            scores.append(value); decisions.append(decision)
        return scores,decisions


def main():
    p=argparse.ArgumentParser(); p.add_argument('--method',required=True,choices=['simmark','pmark-online','pmark-offline','alimark','kgw','sir','semstamp','k-semstamp'])
    p.add_argument('--watermarked',required=True); p.add_argument('--null',required=True); p.add_argument('--out',required=True)
    p.add_argument('--attacks',nargs='+',default=['none','reorder','delete','dilute','substitute'])
    p.add_argument('--strengths',nargs='+',type=float,default=[.5]); p.add_argument('--watermarked-paraphrases'); p.add_argument('--null-paraphrases')
    p.add_argument('--model',default=os.environ.get('MODEL','facebook/opt-1.3b')); p.add_argument('--embedder',default='sentence-transformers/all-mpnet-base-v2')
    p.add_argument('--precision',choices=['fp16','bf16','fp32'],default=None)
    p.add_argument('--vllm-python',default=sys.executable)
    p.add_argument('--vllm-max-model-len',type=int,default=2048)
    p.add_argument('--vllm-max-num-seqs',type=int,default=192)
    p.add_argument('--vllm-gpu-memory-utilization',type=float,default=.80)
    p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=20260820); p.add_argument('--max-documents',type=int); p.add_argument('--alimark-probing',action='store_true'); p.add_argument('--positive-only-attacks',action='store_true'); p.add_argument('--resume',action='store_true'); a=p.parse_args()
    a.out=str((ROOT/a.out).resolve()) if not Path(a.out).is_absolute() else a.out
    wm=rows(a.watermarked); null=rows(a.null); ids=sorted(set(wm)&set(null)); ids=ids[:a.max_documents] if a.max_documents else ids; wpara=external(a.watermarked_paraphrases); npara=external(a.null_paraphrases)
    conditions=[]
    if a.resume and Path(a.out).exists():
        conditions=json.loads(Path(a.out).read_text()).get("conditions", [])
    completed={(row["attack"], float(row["strength"])) for row in conditions}
    detector=Detector(a,[' '.join(null[i]['sentences']) for i in ids])
    score_cache={}
    def cached_scores(texts, prompts, cache_key):
        fingerprint=hashlib.sha256(json.dumps(
            {'texts':texts,'prompts':prompts},ensure_ascii=False,
            separators=(',',':')).encode()).hexdigest()
        if fingerprint not in score_cache:
            score_cache[fingerprint]=detector.score_many(
                texts,cache_key,prompts=prompts)
        return score_cache[fingerprint]
    for kind in a.attacks:
        strengths=[0.] if kind=='none' else a.strengths
        for strength in strengths:
            if (kind, float(strength)) in completed:
                print(f"{a.method}: {kind} {strength:g} already complete", flush=True)
                continue
            wt=[]; nt=[]; wp=[]; nprompts=[]
            pool=[s for i in ids for s in null[i]['sentences']]
            for i in ids:
                wrng=np.random.default_rng(seed((a.seed,i,kind,strength)))
                wt.append(' '.join(alimark_probing_attack(wm[i]['sentences'],kind,strength,wrng) if a.alimark_probing else attack(wm[i]['sentences'],pool,kind,strength,wrng,wpara.get(i))))
                wp.append(str(wm[i].get('prompt','')).strip())
                if a.positive_only_attacks: nt.append(' '.join(null[i]['sentences']))
                else:
                    nrng=np.random.default_rng(seed((a.seed,i,kind,strength)))
                    nt.append(' '.join(alimark_probing_attack(null[i]['sentences'],kind,strength,nrng) if a.alimark_probing else attack(null[i]['sentences'],pool,kind,strength,nrng,npara.get(i))))
                nprompts.append(str(null[i].get('prompt','')).strip())
            tag=f'{kind}-{strength:g}'
            ws,wd=cached_scores(wt,wp,f'{tag}-watermarked')
            ns,nd=cached_scores(nt,nprompts,f'{tag}-null')
            if wd is None:
                threshold=sorted(ns,reverse=True)[0]; wd=[x>=threshold for x in ws]; nd=[x>=threshold for x in ns]
            conditions.append({'attack':kind,'strength':strength,'watermarked':{'documents':len(ws),'detections':sum(wd),'rate':sum(wd)/len(wd),'mean_score':sum(ws)/len(ws)},'null':{'documents':len(ns),'detections':sum(nd),'rate':sum(nd)/len(nd),'mean_score':sum(ns)/len(ns)},'scores':{'watermarked':ws,'null':ns}})
            atomic_json(a.out,{'experiment':'exp34_native_baseline_robustness','method':a.method,'config':vars(a),'conditions':conditions})
            print(kind,strength,conditions[-1]['watermarked']['rate'],conditions[-1]['null']['rate'],flush=True)
    if getattr(detector, 'llm', None) is not None:
        detector.llm.close()
    os.chdir(detector.cwd)

if __name__=='__main__': main()
