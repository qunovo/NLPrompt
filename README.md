# NLPrompt: Noise-Label Prompt Learning for Vision-Language Models

This is the implementation of our paper [NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](https://arxiv.org/abs/2412.01256). 

![NLPrompt Framework](https://github.com/qunovo/NLPrompt/blob/master/NLPrompt-framework.png?raw=true)

## How to Install

This code is built on top of the [CoOp](https://github.com/KaiyangZhou/CoOp). Please follow their steps to configure the runtime environment. Many thanks for their contributions!

Follow CoOp to install the datasets. Note that the [Food101N](https://www.kaggle.com/datasets/kuanghueilee/food-101n) dataset needs to be downloaded separately. [Food101N]([Food101N](https://www.kaggle.com/datasets/kuanghueilee/food-101n)) uses the same test set as Food101.

## How to Run

We provide the running scripts in `scripts/nlprompt`. Below we provide examples on how to run NLPrompt on Caltech101.

**NLPrompt(Caltech101, Sym)**:

-  `bash scripts/nlprompt/main.sh caltech101 16 0.50 'sym' 100`

**NLPrompt(Caltech101, Asym)**:

-  `bash scripts/nlprompt/main.sh caltech101 16 0.50 'asym' 100`

