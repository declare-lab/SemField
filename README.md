# 🌐 SemField

**A Simple, Linear, Continuous, yet Robust Semantic Watermark**

Code for SemField and its polarity-locked variant, SemField-PL, together with
generation, detection, robustness, and text-quality evaluation tools.

## 📖 Overview

> SemField watermarks text by selecting sentences in a continuous semantic
> embedding space. A secret key defines a Gaussian field direction. At each
> generation step, the method evaluates candidate sentences using the
> normalized prospective document aggregate and selects the candidate with
> the highest keyed projection. SemField reinforces the positive direction.
> SemField-PL chooses either direction from the first candidate set and fixes
> that orientation for the document. Detection recomputes sentence embeddings
> and evaluates their normalized aggregate. Under an ideal Gaussian key
> independent of the document, the signed statistic has a standard Gaussian
> null distribution. The aggregate is invariant to sentence reordering when
> the sentences and their embeddings are preserved.

## 🗂️ What Is Here

```text
.
├── semfield/          # semantic fields, feature maps, encoders, and inference
├── experiments/       # generation, detection, attacks, calibration, and quality
├── third_party/       # baseline implementations and upstream licenses
├── pyproject.toml     # package metadata and optional dependencies
└── LICENSE            # MIT license for SemField code
```


## ⚙️ Setup

Python 3.10 or newer is required. Generation uses vLLM and a compatible CUDA
environment. Install generation and embedding dependencies with `uv`:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[embed,generation]'
```

For numerical detection alone, install `uv pip install -e .`. Additional
extras are `baselines` for numerical baseline dependencies and `quality` for
MAUVE and language-model evaluation:

```bash
uv pip install -e '.[baselines,quality]'
```

Models are loaded from Hugging Face. First use requires network access or a
populated cache. Gated models require accepting their terms and authenticating
with Hugging Face. Set `HF_HOME` to choose the model cache and
`SEMFIELD_EMBCACHE` to choose the sentence-embedding cache.

## 🚀 Generate Watermarked Text

Save a prompt manifest as `prompts.json`:

```json
{
  "documents": [
    {"document_id": 0, "prompt": "The city announced a new public transport plan."},
    {"document_id": 1, "prompt": "Scientists reported new findings about ocean ecosystems."}
  ]
}
```

Each record needs a unique integer `document_id` and a string `prompt`.
An optional `reference` field can store its human-reference continuation.

### SemField

```bash
python experiments/exp19_semfield_generate.py \
  --input prompts.json \
  --generator opt-1.3b --embedder mpnet \
  --watermark true \
  --polarity-locked false \
  --documents 2 --sentences 12 --candidates 64 \
  --temperature 0.7 --top-p 0.95 --repetition-penalty 1.05 \
  --seed 0 --key-seed 0 \
  --out-dir generations
```

### SemField-PL

```bash
python experiments/exp19_semfield_generate.py \
  --input prompts.json \
  --generator opt-1.3b --embedder mpnet \
  --watermark true \
  --polarity-locked true \
  --documents 2 --sentences 12 --candidates 64 \
  --temperature 0.7 --top-p 0.95 --repetition-penalty 1.05 \
  --seed 0 --key-seed 0 \
  --out-dir generations-pl
```

Use `--polarity-locked false` for SemField (the default) and
`--polarity-locked true` for SemField-PL. Both commands write
`opt-1.3b_semfield.json` and checksummed feature sidecars.
Keep the JSON files and feature sidecars together.

Watermarking is enabled by default. Use `--watermark false` to generate only
unwatermarked model text, saved as `opt-1.3b_none.json`. In this mode,
watermark-specific options such as `--polarity-locked`, `--candidates`, and
`--key-seed` are ignored. Temperature, token budgets, and other ordinary
generation settings still apply. To obtain both outputs for an experiment,
run the command separately with `--watermark true` and `--watermark false`.

Other configured generators include `llama3.2-1b` and `qwen3-1.7b`.
See [generation.py](semfield/generation.py) and
[embedders.py](semfield/embedders.py) for supported identifiers.
Use `--help` for sampling, batching, and device options. The worker uses the
active Python interpreter unless `--vllm-python` is supplied.

`--key-seed` defines a reproducible benchmark key. Public benchmark seeds are
not deployment secrets. Use private key management for deployment.

## 🔍 Detection

The Python API accepts a matrix of sentence features. This small numerical
example demonstrates both tail conventions:

```python
import numpy as np

from semfield.semfield import KeyedSemanticField, two_sided_field_statistic
from semfield.semfield_kernels import LinearSphericalFeatures

features = np.array([[1.0, 0.0], [0.0, 1.0]])
detector = KeyedSemanticField(LinearSphericalFeatures(2), b"example-private-key")

regular = detector.detect_features(features)
polarity_locked = two_sided_field_statistic(features, detector.weights)
print(regular.z_score, regular.analytic_pvalue)
print(polarity_locked.z_score, polarity_locked.analytic_pvalue)
```

For real text, use the same encoder, feature map, and key as generation.
The returned statistic includes the score, aggregate variance, and analytic
Gaussian p-value. Empirical thresholds can be fitted separately using the
calibration tools. The analytic null result assumes an ideal Gaussian key
independent of the tested document.

## 🧪 Evaluate Robustness

Evaluate saved generations under structural edits:

First generate the unwatermarked comparison file with the same prompts,
model, sampling settings, and output directory, using `--watermark false`.
The commands above generate only the watermarked file.

```bash
python experiments/exp20_semfield_robustness.py \
  --watermarked generations-pl/opt-1.3b_semfield.json \
  --null generations-pl/opt-1.3b_none.json \
  --attacks none insert delete reorder dilute \
  --strengths 0.0 0.2 0.4 0.6 0.8 \
  --out robustness.json
```

The evaluator validates saved features and re-embeds edited text. This command
uses the evaluator's analytic Gaussian test at a nominal 1% level. It does not construct the
paper's benchmark ROC tables or independently calibrated thresholds.

Paraphrase generation is provided by
`experiments/exp24_generate_paraphrases.py`. Prepared rewrites can be supplied
to the robustness evaluator through `--watermarked-paraphrases` and
`--null-paraphrases`. Detector-specific null calibration is provided by
`calibrate_baseline_null.py`, `calibrate_external_null.py`, and
`audit_semfield_large_null.py`. Consult each script's `--help` for its input
format and options.

## 📊 Evaluate Text Quality

### Perplexity

```bash
python experiments/exp23_semfield_quality.py \
  --generations 'generations-pl/*.json' \
  --judge meta-llama/Llama-3.1-8B --allow-downloads \
  --out quality.json
```

Select the language-model judge with `--judge`. Use `--skip-ppl` for a quick
surface-statistics pass without loading a judge. Perplexity comparisons should
use the same judge and tokenization across methods.

### MAUVE

```bash
python experiments/eval_text_quality.py \
  --input SemField=generations/opt-1.3b_semfield.json \
  --input SemField-PL=generations-pl/opt-1.3b_semfield.json \
  --reference references.json \
  --seeds 25 26 27 \
  --out mauve.json
```

The reference artifact needs a `documents` list with matching integer
`document_id` values and human-reference continuations in `text`. All inputs
must contain continuation-only text. The evaluator checks exact ID agreement
and matching prompts when present. It reports MAUVE for each seed, their mean,
and sample standard deviation. Higher MAUVE indicates greater distributional
similarity to the supplied references.

## 🧩 Baselines

The experiment scripts provide generation and evaluation adapters for token
and semantic watermark baselines. Relevant entry points include:

| Workflow | Entry point |
| --- | --- |
| SemStamp / k-SemStamp generation | `exp16_semstamp_generate.py` / `exp25_ksemstamp_generate.py` |
| MarkLLM baselines | `exp27_markllm_baseline.py` / `exp38_markllm_semantic_shared.py` |
| External semantic baseline inputs | `exp93_prepare_external_semantic_inputs.py` |
| External semantic baseline scoring | `exp102_external_unified_scores.py` |

These scripts are in `experiments/`. Baseline checkpoints and learned assets
must be obtained separately. Some workflows also require upstream dependencies
outside the core package.

## 📚 Citation

If you use SemField in your research, please cite:

```bibtex
@misc{gumma2026semfieldsimplelinearcontinuous,
      title={SemField: A Simple, Linear, Continuous, yet Robust Semantic Watermark},
      author={Varun Gumma and Navonil Majumdar and Soujanya Poria},
      year={2026},
      eprint={2610.11848},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2610.11848},
}
```
