<div align="center">
  <img src="assets/Molmo2-logo.svg" alt="Molmo2 Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>Teach a Molmo2Fish: Towards interactive fish tracking with natural language guidance</h1>
</div>
<p align="center">
  <a href="https://huggingface.co/tidalove/Molmo2Fish">
    <img alt="Model Checkpoint" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-yellow">
  </a>
  <a href="https://huggingface.co/datasets/tidalove/cfc-track-instruction">
    <img alt="Molmo2Fish Datasets" src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Dataset-yellow">
  </a>
</p>

Video object tracking is important to various population monitoring, behavioral analysis, and wildlife management use cases in ecology. But dealing with tracks in video manually—whether annotating new tracks in video, or modifying existing/predicted tracks to make data ready for downstream analysis—still requires lots of human time and labor.

<strong>Could a model with sufficient intelligence in both <em>natural language and spatiotemporal reasoning</em> help make targeted edits to tracks in video, leveraging the user's domain knowledge without the tedious process of manually annotating every frame?</strong>

We trained Molmo2Fish to probe this question by fine-tuning [Molmo2](https://github.com/allenai/molmo2), Ai2's open video understanding, pointing, and tracking model, on thousands of [fish tracking](https://huggingface.co/datasets/perona-lab/cfc26) correction trajectories. Molmo2Fish treats the track correction task as a conversation in which it incorporates the user's feedback as natural language.

<video src="https://github.com/user-attachments/assets/1dfb98e2-33d2-43d9-8d68-74ea858a8ea1" controls width="700"></video>

<details>
<summary><strong>More video examples</strong></summary>

<video src="https://github.com/user-attachments/assets/7799836a-a614-42a7-8e4b-dbea732d8633" controls width="700"></video>
<video src="https://github.com/user-attachments/assets/cc429865-7538-4c80-9586-f21a4abe001b" controls width="700"></video>
<video src="https://github.com/user-attachments/assets/650dde9e-ef2d-4553-bd3c-f07e0e24b973" controls width="700"></video>
<video src="https://github.com/user-attachments/assets/e934bee6-4022-40f3-84e6-7c994cf3de37" controls width="700"></video>

</details>

**Can we adapt Molmo2 to a visual domain far outside its training regime?** ***Yes.*** LoRA finetuning on the fish tracking task brings Molmo2's performance from 5% tracking accuracy to 79% tracking accuracy.

**Can we teach Molmo2 to understand the track correction task—and more broadly, the skill of referring back to its previous prediction—even though this wasn't part of its original training data?** ***Yes.*** Versions of Molmo2 trained without our track correction data don't perform well on the track correction task (even if they've seen the fish sonar data and perform well at pure tracking), while LoRA finetuning on the correction task brings significant gains over the pre-correction step.

**Can Molmo2Fish use language guidance to intelligently correct flawed predictions?** ***Sometimes***. We probe this question by comparing the corrections Molmo2Fish makes in response to a generic prompt like "Fix all mistakes" to the corrections it makes in response to a detailed prompt specifying all the mistakes. (During training, it sees both variants of the task.) Molmo2Fish is able to correct the "easiest" mistakes regardless of whether the fixes are specified, and unable to correct the "hardest" mistakes regardless of whether the fixes are specified. Only for mistakes falling into an "intermediate" range of difficulty is language guidance necessary and sufficient for Molmo2Fish to make a better correction based on language.

We train Molmo2Fish on a variety of data. A **synthetic** correction set artificially applies corruptions to ground truth tracks and asks the model to correct them. A **targeted** set artificially applies corruptions and asks the model to correct only one or two mistakes. **Molmo-low** and **Molmo-high** represent correction trajectories on predictions from a low-performing (early checkpoint) and high-performing (late checkpoint) version of Molmo2, respectively. Finally, the two-stage tracking-by-detection **YOLO+SORT** pipeline produces correction trajectories used only in evaluation.

<img src="assets/hota_before_after_anim.gif" alt="HOTA before/after correction" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>

Each point here represents one correction trajectory: one pre-correction set of tracks on a video (whether synthetically corrupted, or produced via inference with Molmo or YOLO), whose track accuracy vs. ground truth is on the x axis; and Molmo2Fish's output, or the post-correction tracks, whose track accuracy is on the y axis. (Only the "targeted" evaluation has a different ground truth from the rest, according to the selected mistakes the model was asked to make, which is why a generic "fix all mistakes" prompt *degrades* performance on the "targeted" pre-correction baseline.)

See [our paper](https://arxiv.org/abs/2608.18602) for details, ablations, a description of the data generation pipeline, more correction examples, and more experiments. The rest of this README is focused on our code.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
  - [Installation](#installation)
  - [Downloading Data](#downloading-data)
  - [Environment](#environment)
- [Training and Evaluations](#training-and-evaluations)
  - [Checkpoints](#checkpoints)
  - [Fine-tuning](#fine-tuning)
  - [Evaluation](#evaluation)
    - [Install Vision Process Package](#install-vision-process-package)
    - [Install vLLM](#install-vllm)
    - [Convert Checkpoint to Hugging Face Format](#convert-checkpoint-to-hugging-face-format)

# Overview

This repository is an extension of Ai2's open vision language model, Molmo2, to the tasks of fish tracking and track correction. It preserves all the functionality of Ai2's original Molmo2 release. Our new contributions, described in [our paper](https://arxiv.org/abs/2608.18602) and implemented in this repo, are:

- LoRA fine-tuning
- Training and evaluation on track correction tasks
- Utilities specific to the [Caltech Fish Counting](https://huggingface.co/datasets/perona-lab/cfc26) dataset

Molmo2 is state-of-the-art among open-source models and demonstrates exceptional new capabilities in point-driven grounding in video tasks. This README is written to be self-contained but does not include the information on different training stages and validation on the original, full Molmo2 video data corpus that is detailed in [Ai2's repository](https://github.com/allenai/molmo2). Instead, we focus on a full walkthrough of the new contributions listed above.

# Setup
## Installation
We recommend using python >= 3.11; the package itself requires >= 3.10.
First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system.

To install dependencies, run:

```bash
git clone https://github.com/tidalove/molmo2fish.git
cd molmo2fish
pip install torchcodec
pip install -e .[all]
```

To install dependencies including vLLM (optional but strongly recommended for evaluation), you can run the following or see [Install vLLM](#install-vllm) for more details:

```bash
git clone https://github.com/tidalove/molmo2fish.git
cd molmo2fish
pip install uv
uv pip install "vllm>=0.15.0,<0.24" --torch-backend=auto
pip install torchcodec
pip install -e .[all]
pip install --no-cache-dir "molmo-utils[torchcodec]"
```

`torchcodec` is a required dependency either way, but it's worth installing first and on its
own: it has some complex dependencies that can break when resolved together with everything
else in `install -e .[all]`.

You also need `ffmpeg` on your `PATH`, built with libx264. The videos used for our training and evaluation were encoded with ffmpeg 4.4.2, the Ubuntu 22.04 system package.


## Downloading Data

Our data are stored and described in more detail on [huggingface](https://huggingface.co/datasets/tidalove/cfc-track-instruction) and can be downloaded with the following command. Make sure to set `MOLMO_DATA_DIR` before downloading. The same script handles the download of most other Molmo2 datasets as well.

```bash
export MOLMO_DATA_DIR=./data/
python -m scripts.download_datasets cfc --n-procs 8
```

Downloading can be resumed if canceled or an error occurs mid-download.

**Note:** The tracking task on this dataset is *not equivalent* to the tracking task on the original CFC dataset. Compared to the videos in the original CFC dataset release, we subsample frames in time by a factor of 3: we encode videos at 6fps and construct the tracking task at 2fps, in line with the Molmo2 video tracking approach, for the sake of simplicity and to reuse existing utils when possible. We also slice many videos from their full lengths to shorter clips, some overlapping with each other, in order to fit both tracking and track correction tasks within the model's context window on our GPUs. All metrics in our paper, including metrics from the traditional two-stage tracking-by-detection YOLO+SORT pipeline, are reported on this modified video set, and *not* on the original CFC videos.

## Environment
Generally training runs should use these flags:

```
HF_DATASETS_OFFLINE=1
OLMO_SHARED_FS=1
HF_ACCESS_TOKEN=YOUR_HF_KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
WANDB_API_KEY=YOUR_WANDB_KEY
WANDB_PROJECT=YOUR_WANDB_PROJECT
WANDB_ENTITY=YOUR_WANDB_ENTITY
OMP_NUM_THREADS=8
```

`HF_DATASETS_OFFLINE` stops HF from sending tons of requests to the HF dataset hub even though the data
is already download.

`OLMO_SHARED_FS` tell the codes to assume, for multi-nodes jobs, you are saving to a shared
file system.

`HF_ACCESS_TOKEN` might be used to download the tokenizer, `OPENAI_API_KEY` might be used in some evaluations, 
and `WANDB_API_KEY` is for wandb logging.

`WANDB_PROJECT` is what actually switches wandb on: `launch_scripts/sft.py` disables wandb
entirely when it is unset, and any `--wandb.*` override then fails during config resolution.
Set `WANDB_PROJECT` and `WANDB_ENTITY` if you want logging, and leave them unset if you don't.

`OMP_NUM_THREADS` is for torch.

# Training and Evaluations

## Checkpoints

We release the Molmo2Fish weights at [tidalove/Molmo2Fish](https://huggingface.co/tidalove/Molmo2Fish). 
The repo holds the HuggingFace-format weights, ready for evaluation by vLLM and `launch_scripts/hf_eval.py`,
and a tar of the raw training checkpoint (model + optimizer state) for resuming fine-tuning.

```bash
# HF format, ready to evaluate
hf download tidalove/Molmo2Fish --exclude "*.tar" --local-dir Molmo2Fish-HF/step420-hf

# raw weights
hf download tidalove/Molmo2Fish Molmo2Fish-step420-raw.tar --local-dir .
tar -xf Molmo2Fish-step420-raw.tar
```

This is the link to the Molmo2-8B checkpoint that we trained from.
Download and untar it to use as a fine-tuning starting point:

```bash
wget https://storage.googleapis.com/oe-training-public/Molmo2-1225/Molmo2-8B.tar
tar -xf Molmo2-8B.tar
```

For earlier checkpoints of Molmo2-4B, 7B, and 8B after various training stages, see the original Molmo2 [repo](https://github.com/allenai/molmo2).

## Fine-tuning
Multitask training can be done with `launch_scripts/sft.py`. You must specify the training mixture and the model checkpoint to start from.

Fully fine-tuning on just the tracking task, with wandb logging:
```bash
WANDB_API_KEY=key WANDB_PROJECT=project WANDB_ENTITY=entity \
  torchrun --nproc-per-node=8 \
  launch_scripts/sft.py /path/to/pretrained/model cfc_track \
  --wandb.name=run_name \
  --save_folder=/path/to/save/folder
```

`WANDB_PROJECT` and `WANDB_ENTITY` have to be in the environment: wandb is switched off
when `WANDB_PROJECT` is unset.

LoRA fine-tuning on the track correction task, without wandb logging. This is the recipe the
released Molmo2Fish checkpoint used:

```bash
torchrun --nproc-per-node=8 launch_scripts/sft.py /path/to/pretrained/model cfc_correction \
  --lora_llm --lora_vit --lora_connector --lora_rank 64 \
  --save_folder=/path/to/save/folder
```

Freeze a component explicitly to leave it out
of training entirely. For example, LoRA on the LLM, a fully frozen ViT, and a fully
fine-tuned connector:

```bash
torchrun --nproc-per-node=8 launch_scripts/sft.py /path/to/pretrained/model cfc_correction \
  --lora_llm --lora_rank 64 --freeze_vit \
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

### Install vLLM
Molmo2 is officially supported in vLLM starting from v0.15.0, so install
`vllm>=0.15.0,<0.24`. Our evaluations were run on vLLM 0.17.0 with
transformers 4.57.6.

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

The Molmo2Fish download above is already in HF format and can be pointed to directly, no
conversion necessary. The Molmo2-8B tar holds raw training weights, so convert it first if
you want to evaluate the base model.

Evaluation can be done with the `launch_scripts/hf_eval.py` script.

To eval on a single task:

```bash
python launch_scripts/hf_eval.py /path/to/model/hf/checkpoint task_name
```

E.g. to eval our Molmo2Fish checkpoint on the Molmo-low validation subset, with fully detailed correction prompts:

```bash
python launch_scripts/hf_eval.py Molmo2Fish-HF/step420-hf cfc_hf_correction_molmo_low_full_eval_2fps
```

This will save `predictions.json` and `metrics.json` under
`results/cfc_hf_correction_molmo_low_full_eval_2fps/validation` (override with `--save_dir`).
A second call re-runs inference from scratch unless you pass `--resume`; overwrite an existing `metrics.json` by passing `--overwrite`.

Scoring is separable from generation. To re-score a `predictions.json` you already have,
with no GPU and no model:

```bash
python launch_scripts/hf_eval.py - cfc_hf_correction_molmo_low_full_eval_2fps \
  --predictions results/cfc_hf_correction_molmo_low_full_eval_2fps/validation/predictions.json \
  --overwrite
```

For correction tasks the metrics include `HOTA_before` (the tracks the model was handed),
`HOTA_after` (what it returned), and `norm_delta_HOTA`, the fraction of the available
headroom it closed, plus a per-river breakdown and the directional net-count error `nMAE`.
Run `python launch_scripts/hf_eval.py --help` for the full set of options.

The original eval script (with our modifications to support the multi-turn conversation format) at `launch_scripts/eval.py` also works, but is much slower and was tested less extensively.

## Evaluation subsets

One eval/train dataset exists per prompt
tier: `full` (every mistake spelled out), `vague` (the same mistakes, tersely),
`wrong_only` (generic, but says mistakes exist), and `no_info` (generic and
non-committal):

| in the paper | dataset names |
|---|---|
| tracking | `cfc_hf_track` (track all fish), `cfc_hf_target` (track a referred subset) |
| targeted | `cfc_hf_synthetic_correction_incomplete` |
| synthetic | `cfc_hf_synthetic_correction_{full,vague,wrong_only,no_info}` |
| Molmo-high | `cfc_hf_correction_molmo_high_{full,vague,wrong_only,no_info}` |
| Molmo-low | `cfc_hf_correction_molmo_low_{full,vague,wrong_only,no_info}` |
| YOLO+SORT | `cfc_hf_correction_yolo_{full,vague,wrong_only,no_info}` (validation only) |
| text-only | `cfc_hf_text` |

Append `_eval_2fps` to any of these to get its evaluation task name. The
[hub config names](https://huggingface.co/datasets/tidalove/cfc-track-instruction)
are the same minus the `cfc_hf_` prefix (e.g. `cfc_correction_molmo_low_full`).

# Citation

```
@misc{molmo2fish,
      title={Teach a Molmo2Fish: Towards interactive fish tracking with natural language guidance}, 
      author={Kai Van Brunt and Justin Kay and Sara Beery},
      year={2026},
      eprint={2608.18602},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.18602}, 
}
```
