# Self-training with Masked Supervised Contrastive Loss for Unknown Intents Detection
This repository is the official implementation of Self-training with Masked Supervised Contrastive
Loss for Unknown Intents Detection (IJCNN2021) by Zijun Liu, Yuanmeng Yan, Keqing He, Sihong Liu, Hong Xu, Weiran Xu.

## Introduction
An iterative learning framework that can dynamically improve the model's ability of OOD intent detection and meanwhile continually obtain valuable new data to learn deep discriminative features. 
![0001.jpg](https://i.loli.net/2021/06/21/YsRhzQPqcwuV8y6.jpg)

## Code
- [back_translate]: for generating the augmentations of samples.  
- [data]: place for storing train/eval/test data.
- [models]: the base model which is Bi-LSTM with supervised contrastive loss.  
- data_process_for_agument.py: python code for processing data which is used during self-training loops.  
- losses.py: defining the masked supervised contrastive loss

## Run
```
bash run.sh  
```
*maybe you need change the data_path or some other hyper-parameters with your settings.* 

