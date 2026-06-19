<div align=center>
<img width="100%" alt="image" src="imgs/teaser.png">
</div>

# CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large Vision-Language Models

<p align="center">
  <a href="https://litelvlm-demo.com/">
    <img src="https://img.shields.io/badge/arXiv-b31b1b?logo=arxiv&logoColor=white">
  </a>
  <a href="https://arxiv.org/abs/2605.13178">
    <img src="https://img.shields.io/badge/🌐%20Project%20Page-Visit-blue">
  </a>
  <a href="https://github.com/sejong-rcv/LiteLVLM">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white">
  </a>
</p>

[Sangin Lee](https://scholar.google.com/citations?hl=ko&view_op=list_works&gmla=AIqSsVtReIryhTuSPRawrTwYvu9NA-xEVwFSQW5_enptxY4xswt0mimpMdlTRMNcsPGnnHCtNuzk0T4gDErqakULVV1Df2sHd23GlASjg1SNvGtkgQ&user=MO_dSVUAAAAJ)$^1$ and [Yukyung Choi](https://scholar.google.com/citations?user=vMrPtrAAAAAJ&hl=en)$^{\ddagger}$

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

## <img src="imgs/logo.png" height="55"> LiteLVLM
In large vision-language models, visual tokens typically constitute the majority of input tokens, leading to substantial computational overhead. To address this, recent studies have explored pruning redundant or less informative visual tokens for image understanding tasks. However, these methods struggle with pixel grounding tasks, where token importance is highly contingent on the input text. Through an in-depth analysis of CLIP, we observe that visual tokens within referent regions often exhibit low similarity to their textual representation. Motivated by this insight, we introduce LiteLVLM, a training-free, text-guided token pruning strategy for efficient pixel grounding inference. By reversing the ranking of CLIP's visual-text similarity, LiteLVLM effectively retains visual tokens covering the referent regions, while recovering context tokens to enable clear foreground-background separation. Extensive experiments demonstrate that LiteLVLM significantly outperforms existing methods by over 5\% across diverse token budgets. Without any training or fine-tuning, LiteLVLM maintains 90\% of the original performance with a 22\% speedup and a 2.3&times; memory reduction.

<div align=center>
<img width="700" alt="intro" src="imgs/intro.png">
</div>

## 🚀 Installation

1. Clone this repository.
```bash
git clone https://github.com/sejong-rcv/LiteLVLM.git
cd LiteLVLM
```

2. Setup a conda environment and install packages.
```bash
conda create -n LiteLVLM python=3.10 -y
conda activate LiteLVLM
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
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
@article{lee2026clip,
  title={CLIP Tricks You: Training-free Token Pruning for Efficient Pixel Grounding in Large VIsion-Language Models},
  author={Lee, Sangin and Choi, Yukyung},
  journal={arXiv preprint arXiv:2605.13178},
  year={2026}
}
```

## 🙏 Acknowledgement
We thank to [GLaMM](https://github.com/mbzuai-oryx/groundingLMM) and [VideoGLaMM](https://github.com/mbzuai-oryx/VideoGLaMM) for releasing their code as open source.
