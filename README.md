# CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models

## 📜 News

🔥 **[2026/05/07]** Our code is now open source!

🔥 **[2026/05/01]** Our LiteLVLM is accepted by **ICML 2026**!

## 📢 Outline

1. [LiteLVLM](#LiteLVLM)
2. [Installation](#Installation)
3. [Preparation](#Preparation)
3. [Model Zoo](#Model-Zoo)
4. [Evaluation](#Evaluation)
5. [License](#License)
6. [Acknowledgement](#Acknowledgement)

## <img src="images/logo.png" height="55"> LiteLVLM
In large vision-language models (LVLMs), visual tokens typically constitute the majority of input sequences, leading to substantial computational overhead. To address this, recent studies have focused on pruning redundant or less important visual tokens for image understanding tasks. However, these methods struggle with pixel grounding, where token importance is highly dependent on the textual input. Through an in-depth analysis of CLIP, we observe that **visual tokens within the object region often exhibit low similarity to the text.** Motivated by this insight, **we propose LiteLVLM, a training-free, text-guided token pruning for efficient pixel grounding inference.** By simply reversing CLIP's visual-text similarity, LiteLVLM retains text-relevant visual tokens essential for grounding, while recovering context tokens to enable clear foreground-background separation. Extensive experiments demonstrate that LiteLVLM significantly outperforms existing methods across diverse token reduction ratios. Without any training or fine-tuning, our LiteLVLM maintains 90\% of the original performance while achieving a 2.2&times; speedup and a 2.3&times; memory reduction. The code will be released on GitHub after publication.

<div align=center>
<img width="700" alt="image" src="images/intro.png">
</div>

## 🚀 Installation

1. Clone this repository.
```bash
git clone https://github.com/sejong-rcv/LiteLVLM.git
cd CLIP-Tricks-You
```

2. Setup a conda environment and install packages.
```bash
conda create -n LiteLVLM python=3.10 -y
conda activate LiteLVLM
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

3. Install mmcv
```bash
git clone https://github.com/open-mmlab/mmcv
cd mmcv
git checkout v1.4.7
MMCV_WITH_OPS=1 pip install -e .
```

## 📌 Datasets

Please see [`docs/datasets.md`](docs/datasets.md) for dataset preparation guidelines.

## 🔍 Model Zoo

We use the official pretrained checkpoints released by [GLaMM](https://github.com/mbzuai-oryx/groundingLMM/blob/main/docs/model_zoo.md).
Download the `GLaMM-RefSeg` from the HugginFace and place it in `checkpoints/`.
If you plan to fine-tune LiteLVLM, please additionally download the `GLaMM-GranD-Pretrained` checkpoint.

## ⚡ Evaluation

Run the following example to evaluate our LiteLVLM on Referring Expression Segmentation benchmarks.

<details>

<summary>
1. Prepare the pretrained checkpoints and datasets.
</summary>
  
- Check [`MODEL_ZOO`](#Model-Zoo) to 
  - download the pretrained pixel grounding model checkpoints to the folder `./checkpoints/`.
- Check [`Datasets`](docs/datasets.md) to set up dataset.

</details>

<details>

<summary>
2. After preparation, run the following script to evaluate LiteLVLM.
</summary>
  
```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=22999

CKPT_PATH=$1
REF_SEG_DATASET=$2
RESULT_PATH=$3
RETAIN_TOKENS=$4

deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py \
    --version "$CKPT_PATH" \
    --refer_seg_data "$REF_SEG_DATASET" \
    --results_path "$RESULT_PATH" \
    --num_retain_tokens $RETAIN_TOKENS \
    --pretrained
```

- To evaluate the **RefCOCO** benchamrk with **192 retained tokens**, run the following command:
```bash
bash eval/referring_seg/single_evaluation.sh 'checkpoints/GLaMM-RefSeg' 'refcoco|val' 'run/LiteLVLM/192' 192
```
</details>

<details>

<summary>
3. One-Click evaluation
</summary>
  
- If you want to evaluate all benchmarks, run the following commad:
```bash
bash eval/referring_seg/run_evaluation.sh 'checkpoints/GLaMM-RefSeg' 'run/LiteLVLM/192' 192
```
</details>

## 📝 License
This project is released under the [Apache 2.0 license](./LICENSE).

## Citation

If you use LiteLVLM in your research, please cite our work by using the following BibTeX entry:
```bibtex

```

## 🙏 Acknowledgement
We thank to [GLaMM](https://github.com/mbzuai-oryx/groundingLMM) and [VideoGLaMM](https://github.com/mbzuai-oryx/VideoGLaMM) for releasing their code as open source.
