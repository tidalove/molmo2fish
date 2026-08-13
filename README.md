<div align="center">
  <img src="assets/Molmo2-logo.svg" alt="Molmo2 Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>Molmo 2: State-of-the-art video understanding, pointing, and tracking</h1>
</div>
<p align="center">
  <a href="https://github.com/allenai/molmo2/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://allenai.org/blog/molmo2">
    <img alt="Blog Post" src="https://img.shields.io/badge/Molmo2-blog-F0529C">
  </a>
  <a href="https://arxiv.org/abs/2601.10611">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2601.10611-blue">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmo2">
    <img alt="Model Checkpoints" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Models-yellow">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmo2-data">
    <img alt="Molmo2 Datasets" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Datasets-yellow">
  </a>
</p>


This repository is an extension of Ai2's open vision language model, Molmo2, for fish tracking and track correction. It preserves all the functionality of Ai2's original Molmo2 release. Our new contributions, described in [our paper]() and implemented in this repo, are:

- LoRA fine-tuning
- Training and evaluation on track correction tasks
- Utilities specific to the [Caltech Fish Counting]() dataset we use as an initial testbed for track correction capabilities

Molmo2 is state-of-the-art among open-source models and demonstrates exceptional new capabilities in point-driven grounding in single image, multi-image, and video tasks as shown below. This README is written to be self-contained but does not include the information on different training stages and validation on the original, full Molmo2 video data corpus that is detailed in [Ai2's repository](). Instead, we focus on a full walkthrough of the new contributions listed above.



## Table of Contents
- [Setup](#setup)
  - [Installation](#installation)
  - [Downloading Data](#downloading-data)
  - [Downloading Pretrained Models](#downloading-pretrained-models)
  - [Visualizing Data](#visualizing-data)
  - [Environment](#environment)
- [Training and Evaluations](#training-and-evaluations)
  - [Checkpoints](#checkpoints)
  - [SFT Training](#sft-training)
  - [Evaluation](#evaluation)
  - [Context Parallel](#context-parallel)
- [Transformers and vLLM](#transformers-and-vllm)
  - [Convert Checkpoint to Hugging Face Format](#convert-checkpoint-to-hugging-face-format)
  - [Transformers Inference](#transformers-inference)
    - [Image Inference Example](#image-inference-example)
    - [Video Inference Example](#video-inference-example)
  - [MolmoPoint Transformers Inference](#molmopoint-transformers-inference)
      - [Image Inference Example](#image-inference-example)
      - [Video Inference Example](#video-inference-example)
  - [Fast Inference with vLLM](#fast-inference-with-vllm)
    - [Install Vision Process Package](#install-vision-process-package)
    - [Install vLLM (\>= 0.15.0)](#install-vllm--0150)
    - [Run vLLM inference (Gradio Demo)](#run-vllm-inference-gradio-demo)
- [Code](#code)
  - [Data Pipeline](#data-pipeline)
  - [Message Trees](#message-trees)
  - [Packing](#packing)

# Setup
## Installation
We recommend using python >= 3.11 
First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system.

To install dependencies, run:

```bash
git clone https://github.com/tidalove/molmo2fish.git
cd molmo2fish
pip install torchcodec
pip install -e .[all]
```

It's recommended to install torchcodec separately since it has some complex dependencies that 
can break if installed in combination with the others as done using `install -e .[all]`


## Downloading Data

Our data are stored and described in more detail on [huggingface](https://huggingface.co/datasets/tidalove/cfc-track-instruction) and can be downloaded with the following command. Make sure to set `MOLMO_DATA_DIR` before downloading. The same script handles the download of most other Molmo2 datasets as well.

```bash
export MOLMO_DATA_DIR=./data/
python -m scripts.download_datasets cfc --n_proc 8
```

Downloading can be resumed if canceled or an error occurs mid-download.

The tracking task on this dataset is *not equivalent* to the tracking task on the original CFC dataset. Compared to the videos in the original CFC dataset release, we subsample frames in time by a factor of 3: we encode videos at 6fps and construct the tracking task at 2fps, in line with the Molmo2 video tracking approach, for the sake of simplicity and to reuse existing utils when possible. We also slice many videos from their full lengths to shorter clips, some overlapping with each other, in order to fit both tracking and track correction tasks within the model's context window on our GPUs. All metrics in our paper, including metrics from the traditional two-stage detection-by-tracking YOLO+SORT pipeline, are reported on this modified video set, and *not* on the original CFC videos.

## Environment
Generally training runs should use these flags:

```
HF_DATASETS_OFFLINE=1
OLMO_SHARED_FS=1
HF_ACCESS_TOKEN=YOUR_HF_KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
WANDB_API_KEY=YOUR_WANDB_KEY
OMP_NUM_THREADS=8
```

`HF_DATASETS_OFFLINE` stops HF from sending tons of requests to the HF dataset hub even though the data
is already download.

`OLMO_SHARED_FS` tell the codes to assume, for multi-nodes jobs, you are saving to a shared
file system.

`HF_ACCESS_TOKEN` might be used to download the tokenizer, `OPENAI_API_KEY` might be used in some evaluations, 
and `WANDB_API_KEY` is for wandb logging.

`OMP_NUM_THREADS` is for torch.

# Training and Evaluations



Molmo2 training has three stages:

1. **Pre-Training** — Train on image captioning, NLP, and image pointing using `launch_scripts/pretrain.py`. Start from pretrained LLM + ViT weights.
2. **SFT** — Multitask supervised fine-tuning on the full mixture (QA, pointing, tracking, video, etc.) using `launch_scripts/sft.py`. Start from a pretrained checkpoint.
3. **Long-Context SFT** — Continue SFT with longer sequences (36k+ tokens, 384 frames) for improved video understanding. Uses the same `launch_scripts/sft.py` with increased `--seq_len`.

Each stage produces a checkpoint that feeds into the next. We release checkpoints at each stage (see below).

## Checkpoints
We release model weights for the official Molmo2Fish model after rank 64 LoRA fine-tuning of Molmo2-8B on all CFC data.
For convenience we also provide the link to the Molmo2-8B checkpoint that we trained from.
For earlier checkpoints of Molmo2-4B, 7B, and 8B after various training stages, see the original Molmo2 [repo]().

To use this checkpoint download it, untar it, and it can be evaluated or used as a starting point for fine-tuning.
It contains both the huggingface format checkpoint and the raw weights.
For example:

```
wget https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-8B.tar
tar -xf Molmo2-8B.tar
```

## Fine-tuning
Multitask training can be done with `launch_scripts/sft.py`. You must specify the training mixture and the model checkpoint to start from.

Fully fine-tuning on just the tracking task:
```bash
WANDB_API_KEY=key torchrun --nproc-per-node=8 launch_scripts/sft.py /path/to/pretrained/model cfc_track \
  --wandb.name=run_name --wandb.entity=entity --wandb.project=project \
  --save_folder=/path/to/save/folder
```

LoRA fine-tuning on the track correction task:

```bash
WANDB_API_KEY=key torchrun --nproc-per-node=8 launch_scripts/sft.py /path/to/pretrained/model cfc_correction \
  --lora_vit --lora_connector --lora --lora_rank 64 \
  --wandb.name=run_name --wandb.entity=entity --wandb.project=project \
  --save_folder=/path/to/save/folder
```

Here `/path/to/pretrained/model` points to a model checkpoint to start from (typically a pretrained model)
and `cfc_track` or `cfc_correction` refers to what training mixture to use.

## Evaluation

We recommend running evaluation using vLLM for faster inference. All evaluation results in our paper were obtained through vLLM evaluation.

### Install Vision Process Package
To run vLLM inference offline, install the Molmo2 vision processing package, `molmo-utils`, which follows the design of `qwen-vl-utils`.
This package loads images and videos and prepares them for use with the Molmo2 HF processor.

Install with:

```bash
pip install --no-cache-dir "molmo-utils[torchcodec]"
```

### Install vLLM (>= 0.15.0)
Molmo2 is officially supported in vLLM starting from v0.15.0.
Please install vLLM 0.15.0 or later.

You can find the detailed installation guide in the [official documentation](https://docs.vllm.ai/en/latest/getting_started/installation).

### Convert Checkpoint to Hugging Face Format
If you have trained a new model or downloaded another Molmo checkpoint, you must first convert the Molmo checkpoint into a HF–compatible format for vLLM inference. You can convert a checkpoint by running:

```bash
# N: 36864 for Molmo2-4B and Molmo2-8B, 65536 for Molmo2-O-7B
python3 -m olmo.hf_model.convert_molmo2_to_hf \
    /path/to/your/checkpoint/dir \
    /path/to/output/dir \
    --override_max_model_len 36864
```

The downloads above include the HF-compatible formats, which can be pointed to directly, no conversion necessary.

Evaluation can be done with the `launch_scripts/hf_eval.py` script.

To eval on a single task:

```bash
python launch_scripts/hf_eval.py /path/to/model/hf/checkpoint task_name
```

E.g. to eval our Molmo2Fish checkpoint on the molmo-low validation subset, with fully detailed correction prompts:

```bash
python launch_scripts/hf_eval.py Molmo2Fish-HF/step300-hf cfc_correction_molmo-low_full_eval_2fps
```

This will save the metrics and predictions under `results/Molmo2Fish/results/cfc_correction_molmo-low_full_eval_2fps/validation`. Future calls to the
eval script will re-use cached metrics if they exist. See `EvalConfig` for additional config options.

The original eval script (with our modifications to support the multi-turn conversation format) at `launch_scripts/eval.py` also works, but is much slower and was tested less extensively.

### Supported Evaluation Groups

TODO: fill this in

| Task name | Benchmarks |
|---|---|
| `single_image` | COCO VQA, TextVQA, ChartQA, DocQA, InfoQA, AI2D, MMMU, RealWorldQA, MathVista, CountBench, PixMo Count, PointingEval v2, PointBench |
| `single_image_test` | Test splits of image tasks + A-OKVQA (MC & DA) |
| `multi_image` | MuirBench, MMIU, BLINK |
| `short_video` | MVBench, TOMATO, MotionBench, TempCompass, PerceptionTest, EgoSchema, NeXTQA |
| `long_video` | VideoMME (w/ and w/o subtitles), LongVideoBench (w/ and w/o subtitles), LVBench, MLVU, VixMo Caps, VideoEvalPro |
| `video` | `short_video` + `long_video` |
| `video_no_subtitle` | `short_video` + long video benchmarks without subtitles |
| `video_subtitle` | Long video benchmarks with subtitles only |
| `video_pointing` | VixMo Points (count & point eval), MeViS point tracking |
| `image_pointing` | CountBench, PixMo Count, PointingEval v2, PointBench |
| `tracking` | MeViS, Ref-YT-VOS, Ref-DAVIS17, ReasonVOS, Molmo2VideoTrack |
| `test_video` | Test-split video benchmarks (MLVU, PerceptionTest, EgoSchema, MotionBench, LongVideoBench) |

Individual task names (e.g. `chart_qa`, `mvbench`, `mevis_track_eval_1fps:test`) can also be passed directly.

### Evaluation Tips

- `NCCL_TIMEOUT_MINUTES=20` can be needed if evaluating long video benchmarks where individual
processes can finish at very different times.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a torch setting that can reduce the chance of OOM errors.
- Memory costs can also be reduced by using the `load_bf16` flag to keep the weights in bfloat16.
We don't use this by default but it generally does not affect performance.
- Both commands can be run with multi-node configuration using `--nnodes` and `--node_rank` as usual with torchrun.

# Citation

```
@article{molmo2fish,
    title={Teach a Molmo2Fish: Towards interactive },
    author={Kai van Brunt and Justin Kay and Sara Beery},
    year={2026},
    journal={arXiv preprint arXiv:2601.10611}
}
```