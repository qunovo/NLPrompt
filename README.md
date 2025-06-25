# NLPrompt: Noise-Label Prompt Learning for Vision-Language Models (CVPR 2025 Highlight)

This is the official PyTorch implementation for the CVPR 2025 highlight paper: [NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2412.01256). 

![NLPrompt Framework](https://github.com/qunovo/NLPrompt/blob/master/NLPrompt-framework.png?raw=true)

## How to Install

Make sure [conda](https://www.anaconda.com/distribution/) is installed properly.

```bash
# Clone this repo
git clone https://github.com/qunovo/NLPrompt.git
cd NLPrompt/Dassl.pytorch

# Create a conda environment
conda create -y -n nlprompt python=3.8

# Activate the environment
conda activate nlprompt

# Install torch and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install this library 
python setup.py develop

cd ..
```

Follow the instructions in [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to prepare the datasets.

Note that the [Food101N](https://www.kaggle.com/datasets/kuanghueilee/food-101n) dataset needs to be downloaded separately. [Food101N]([Food101N](https://www.kaggle.com/datasets/kuanghueilee/food-101n)) uses the same test set as Food101.

This code is built on top of the [CoOp](https://github.com/KaiyangZhou/CoOp) and [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). We sincerely appreciate their contributions!

## How to Run

We provide the running scripts in `scripts/nlprompt`. 

You need to make sure the data path is correct before you run it.

Below we provide examples on how to run NLPrompt on the Caltech101 dataset:

**NLPrompt (Caltech101, Sym)**:

-  `bash scripts/nlprompt/main.sh caltech101 16 0.50 'sym' 100`

**NLPrompt (Caltech101, Asym)**:

-  `bash scripts/nlprompt/main.sh caltech101 16 0.50 'asym' 100`

where the first parameter is the name of the dataset, the second parameter is the number of shots, the third parameter is the noise rate, the fourth parameter is the type of noise and the last parameter is the number of categories of the dataset.

After the experiments, all the results are saved to `output/`.

## Citation

If you find our work useful in your research, please consider citing it!

```
@inproceedings{pan2025nlprompt,
  title={NLPrompt: Noise-Label Prompt Learning for Vision-Language Models},
  author={Pan, Bikang and Li, Qun and Tang, Xiaoying and Huang, Wei and Fang, Zhen and Liu, Feng and Wang, Jingya and Yu, Jingyi and Shi, Ye},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19963--19973},
  year={2025}
}
```

