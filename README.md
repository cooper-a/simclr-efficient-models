# Self-Supervised Contrastive Learning for Efficient Models

Cooper Ang

Accomanied Paper is availiable as a PDF [here](Self_Supervised_Contrastive_Learning_for_Efficient_Models.md).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- [PyTorch](https://pytorch.org)
- [CUDA - Nvidia GPU](https://developer.nvidia.com/cuda-11-7-1-download-archive)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
- thop
- matplotlib
- numpy
- pandas
```
pip install thop
pip install matplotlib
pip install numpy
pip install pandas
```

## Description
This repository contains the code for the final project of SYDE 572. The goal of this project is to implement a self-supervised contrastive learning algorithm and apply it with smaller more efficient model architectures. The code is based on the [SimCLR](https://arxiv.org/abs/2002.05709) paper.

## Dataset
`CIFAR10` dataset is used in this repo, the dataset will be downloaded into `data` directory by `PyTorch` automatically.

## Usage
### Train SimCLR
```
python main.py
optional arguments:
--feature_dim                 Feature dim for latent vector [default value is 128]
--temperature                 Temperature used in softmax [default value is 0.5]
--k                           Top k most similar images used to predict the label [default value is 200]
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 500]
--model_base                  The base model used for training [default value is 'mobilenetv3_small'] Options are 'mobilenetv3_small', 'mobilenetv3_large', 'resnet18'
```

### Linear Evaluation
```
python linear.py
optional arguments:
--model_path                  The pretrained model path [default value is 'results/128_0.5_200_512_500_model.pth']
--batch_size                  Number of images in each mini-batch [default value is 512]
--epochs                      Number of sweeps over the dataset to train [default value is 100]
--model_base                  The base modle used for training [default value is 'mobilenetv3_small'] Options are 'mobilenetv3_small', 'mobilenetv3_large', 'resnet18'
```

### Plotting Results
```
Details are in the `Generate Plots.ipynb` notebook
```

## Results
Detailed results can be found in `final_results` directory.
Nested inside the directory corresponds to different model candidates with self-explanatory titles corresponding to the base model architecture and the hyperparameters used in the training.


## Attribution
This repo is based on the following repo which implemented the SimCLR algorithm:

https://github.com/leftthomas/SimCLR
