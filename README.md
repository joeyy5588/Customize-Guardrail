# Customize Multi-modal RAI Guardrails with Precedent-based Predictions
**Authors**: Cheng‑Fu Yang, Thanh Tran, Christos Christodoulopoulos, Weitong Ruan, Rahul Gupta, Kai‑Wei Chang

[![ArXiv](https://img.shields.io/badge/arXiv-2507.20503-b31b1b.svg)](https://arxiv.org/abs/2507.20503)

🏆 **Accepted to COLM 2025**

## 📖 Overview

This repository contains code and resources for the paper *Customize Multi-modal RAI Guardrails with Precedent-based Predictions*, accepted at **COLM 2025**. 

Building effective multi-modal guardrails requires handling user-defined policies that vary widely and often come with few or no examples. Traditional fine-tuning methods bind models to fixed policies and demand costly retraining to support new or evolving standards, while training-free approaches struggle to embed numerous policies within limited context windows. To overcome these challenges, we condition model judgments on precedents—reasoning traces from prior, similar examples—rather than on static policy definitions. Our framework includes:

- **Critique–Revise Precedent Collection**: An iterative mechanism to generate and refine high-quality precedents that capture policy intent.

- **Precedent-based Inference**: Leverage retrieval-augmented generation (RAG) to integrate precedents into model predictions for robust, flexible filtering.

## 🚀 Installation

```
conda create -n prae python=3.10
git clone -b v0.2.13 https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"

# Install FlashInfer CUDA kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
pip install tabulate open-clip-torch

pip install git+https://github.com/openai/CLIP.git
```

## 📂 Dataset
We are using *UnsafeBench* which is hosted on Huggingface. Please apply for your access here: [link](https://huggingface.co/datasets/yiting/UnsafeBench)

## ⚙️ Launch the LLaVA server
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-13b --tokenizer-path llava-hf/llava-1.5-13b-hf --port 30000 --tp 2
```

## 🧪 Run the experiment
```
python run_experiment.py
```
Use these flags in `run_experiment.py`:

**General arguments**

- `--answer-file` : Path to save model responses (default: logs/llava-prae-16.jsonl).

- `--image-folder` : Directory of input images (default: ./images/test).

- `--temperature` : Sampling temperature for generation (default: 0.0).

- `--max-tokens` : Max tokens in generated output (default: 256).

**Precedent arguments**

- `--num-sample-per-cat` : Precedents sampled per policy category (default: 16).

- `--precedent-model` : Model for generating/embedding precedents (default: llava).

- `--precedent-reflection` : Enable critique–revise loop (default: True).

- `--precedent-file-dir` : Directory of the output precedent files (default: rules/).

- `--cached-precedent-file` : Path to load cached precedents. If specified, precedent collection will be skipped (default: None).

**Retrieval arguments**

- `--retrieval-model` : Embedding model for retrieval (default: facebook/contriever).

- `--retrieval-device` : Device for retrieval model (default: cuda:0).

- `--cached-retrieval-file` : Path to load retrieval results. If specified, precedent collection will be skipped (default: None).

- `--retrieval-th` : Similarity threshold for filtering relevant precedents (default: 0.7).

**Inference arguments**

- `--cached_caption_file` : Path to load/store image captions for speed up inference (default: ``).

- `--inference_model` : Model for final inference (default: llava).

- `--setting` : Evaluation mode: all or subset (default: all).

- `--result-file-dir` : Directory for saving results (default: results/).

### Re-FT
We use the LoRA-finetuning script in the original LLaVA repo: https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh

Please use the data generation script `generate_training_data_reflection.py` to prepare the precedent data in the finetuning format.


## Citation
```
@article{yang2025customize,
  title={Customize Multi-modal RAI Guardrails with Precedent-based predictions},
  author={Yang, Cheng-Fu and Tran, Thanh and Christodoulopoulos, Christos and Ruan, Weitong and Gupta, Rahul and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2507.20503},
  year={2025}
}
```# Customize Multi-modal RAI Guardrails with Precedent-based Predictions
**Authors**: Cheng‑Fu Yang, Thanh Tran, Christos Christodoulopoulos, Weitong Ruan, Rahul Gupta, Kai‑Wei Chang

[![ArXiv](https://img.shields.io/badge/arXiv-2507.20503-b31b1b.svg)](https://arxiv.org/abs/2507.20503)

🏆 **Accepted to COLM 2025**

## 📖 Overview

This repository contains code and resources for the paper *Customize Multi-modal RAI Guardrails with Precedent-based Predictions*, accepted at **COLM 2025**. 

Building effective multi-modal guardrails requires handling user-defined policies that vary widely and often come with few or no examples. Traditional fine-tuning methods bind models to fixed policies and demand costly retraining to support new or evolving standards, while training-free approaches struggle to embed numerous policies within limited context windows. To overcome these challenges, we condition model judgments on precedents—reasoning traces from prior, similar examples—rather than on static policy definitions. Our framework includes:

- **Critique–Revise Precedent Collection**: An iterative mechanism to generate and refine high-quality precedents that capture policy intent.

- **Precedent-based Inference**: Leverage retrieval-augmented generation (RAG) to integrate precedents into model predictions for robust, flexible filtering.

## 🚀 Installation

```
conda create -n prae python=3.10
git clone -b v0.2.13 https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install -e "python[all]"

# Install FlashInfer CUDA kernels
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
pip install tabulate open-clip-torch

pip install git+https://github.com/openai/CLIP.git
```

## 📂 Dataset
We are using *UnsafeBench* which is hosted on Huggingface. Please apply for your access here: [link](https://huggingface.co/datasets/yiting/UnsafeBench)

## ⚙️ Launch the LLaVA server
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.5-13b --tokenizer-path llava-hf/llava-1.5-13b-hf --port 30000 --tp 2
```

## 🧪 Run the experiment
```
python run_experiment.py
```
Use these flags in `run_experiment.py`:

**General arguments**

- `--answer-file` : Path to save model responses (default: logs/llava-prae-16.jsonl).

- `--image-folder` : Directory of input images (default: ./images/test).

- `--temperature` : Sampling temperature for generation (default: 0.0).

- `--max-tokens` : Max tokens in generated output (default: 256).

**Precedent arguments**

- `--num-sample-per-cat` : Precedents sampled per policy category (default: 16).

- `--precedent-model` : Model for generating/embedding precedents (default: llava).

- `--precedent-reflection` : Enable critique–revise loop (default: True).

- `--precedent-file-dir` : Directory of the output precedent files (default: rules/).

- `--cached-precedent-file` : Path to load cached precedents. If specified, precedent collection will be skipped (default: None).

**Retrieval arguments**

- `--retrieval-model` : Embedding model for retrieval (default: facebook/contriever).

- `--retrieval-device` : Device for retrieval model (default: cuda:0).

- `--cached-retrieval-file` : Path to load retrieval results. If specified, precedent collection will be skipped (default: None).

- `--retrieval-th` : Similarity threshold for filtering relevant precedents (default: 0.7).

**Inference arguments**

- `--cached_caption_file` : Path to load/store image captions for speed up inference (default: ``).

- `--inference_model` : Model for final inference (default: llava).

- `--setting` : Evaluation mode: all or subset (default: all).

- `--result-file-dir` : Directory for saving results (default: results/).

### Re-FT
We use the LoRA-finetuning script in the original LLaVA repo: https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_lora.sh

Please use the data generation script `generate_training_data_reflection.py` to prepare the precedent data in the finetuning format.


## Citation
```
@article{yang2025customize,
  title={Customize Multi-modal RAI Guardrails with Precedent-based predictions},
  author={Yang, Cheng-Fu and Tran, Thanh and Christodoulopoulos, Christos and Ruan, Weitong and Gupta, Rahul and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2507.20503},
  year={2025}
}
```