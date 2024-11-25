# NLPrompt: Noise-Label Prompt Learning for Vision-Language Models

![NLPrompt Framework](https://raw.githubusercontent.com/qunovo/images/main/NLPrompt-framework.png?token=AW4XMJHFQDTA552W5HC4JBLHIRP4S)

## How to Install

This code is built on top of the [CoOp](https://github.com/KaiyangZhou/CoOp). Please follow their steps to configure the runtime environment. Many thanks for their contributions!

Follow CoOp to install the datasets. Note that the [Food101N](https://www.kaggle.com/datasets/kuanghueilee/food-101n) dataset needs to be downloaded separately. [Food101N]([Food101N](https://www.kaggle.com/datasets/kuanghueilee/food-101n)) uses the same test set as Food101.

## How to Run

We provide the running scripts in `scripts/nlprompt`. Below we provide examples on how to run NLPrompt on Caltech101.

**NLPrompt(Caltech101, Sym)**:

-  `bash scripts/nlprompt/main.sh caltech101 16 0.50 'sym' 100`

**NLPrompt(Caltech101, Asym)**:

-  `bash scripts/nlprompt/main.sh caltech101 16 0.50 'asym' 100`

