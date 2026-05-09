# Medical LLM Fine-tuning with LoRA and QLoRA

Parameter-efficient fine-tuning of **TinyLlama-1.1B-Chat** for medical question answering, comparing **LoRA** (16-bit base) and **QLoRA** (4-bit base) on the same task.



## Overview

Fine-tunes a small open-source LLM on the medical domain using two parameter-efficient techniques and compares their outputs:

1. **LoRA** (Low-Rank Adaptation) — base model in fp16, trains low-rank adapter matrices on attention projections.
2. **QLoRA** (Quantized LoRA) — base model loaded in **4-bit NF4** with double quantization, trains the same LoRA adapters on top.

Both methods are run with identical hyperparameters so the comparison is fair.

## Dataset

[MedQuAD](https://www.kaggle.com/datasets/jpmiller/layoutlm) — ~47,000 medical question-answer pairs from 12 NIH sources (cancer.gov, MedlinePlus, GARD, etc.).

For Colab free-tier training, a random subsample of 2,000 examples is used (1,950 train / 50 eval). Filtering removes empty rows and entries shorter than 10 characters or longer than 1,500.

## Method

| Component                  | Value                                            |
|----------------------------|--------------------------------------------------|
| Base model                 | TinyLlama/TinyLlama-1.1B-Chat-v1.0               |
| Adapter type               | LoRA (r=8, α=16, dropout=0.05)                   |
| Adapter targets            | `q_proj`, `k_proj`, `v_proj`, `o_proj`           |
| Optimizer                  | AdamW (lr=2e-4, cosine schedule, 3% warmup)      |
| Batch size                 | 4 × grad-accum 4 = effective 16                  |
| Sequence length            | 512                                              |
| Epochs                     | 1                                                |
| Hardware                   | Google Colab free tier (T4 GPU)                  |

| Method | Base precision         | Peak VRAM (TinyLlama-1.1B) |
|--------|------------------------|----------------------------|
| LoRA   | fp16                   | ~5–6 GB                    |
| QLoRA  | NF4 (4-bit) + dq       | ~2–3 GB                    |

## Results

Three model variants generate answers for the same medical questions:

- **Baseline** — un-tuned TinyLlama
- **LoRA** — fp16 base + LoRA adapter
- **QLoRA** — 4-bit base + LoRA adapter

Generated samples are saved to `comparison.md` and shown in `report.pdf`.

## Repository structure

```
.
├── llm_lora_qlora_finetuning.ipynb   # Main notebook
├── README.md
├── report.pdf                        # Submission report
├── comparison.md                     # Side-by-side generations
├── lora_adapter/                     # Trained LoRA adapter (~10 MB)
└── qlora_adapter/                    # Trained QLoRA adapter (~10 MB)
```

## How to run

1. Open `llm_lora_qlora_finetuning.ipynb` in Google Colab.
2. Runtime → Change runtime type → **GPU (T4)**.
3. Get a Kaggle API token (kaggle.com → Settings → API → Create New Token, legacy `kaggle.json` format).
4. Run all cells. Upload `kaggle.json` when prompted.

End-to-end runtime: ~30–40 minutes on a free T4.

## Loading a saved adapter for inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./qlora_adapter")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

## Dependencies

```
transformers>=4.44
peft>=0.12
trl>=0.11
accelerate>=0.33
bitsandbytes>=0.43
datasets>=2.20
kagglehub
```

All installed automatically by the first cell of the notebook.

## Disclaimer

This model is fine-tuned for an academic exercise. It is **not** a medical device and must not be used for actual clinical decision-making. Outputs may contain factual errors or omissions.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023.
- Ben Abacha & Demner-Fushman, *A Question-Entailment Approach to Question Answering* (MedQuAD), BMC Bioinformatics 2019.
- Zhang et al., *TinyLlama: An Open-Source Small Language Model*, 2024.

## Author

_Mahafuj_ · _09.05.26_
