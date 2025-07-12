## HCViT-Net

This is the Pytorch/PytorchLightning implementation of the paper:

* HCViT-Net: Hybrid CNN and Multi Scale Query
Transformer Network for Dermatological Image
Segmentation.

## Introduction

In the computer-aided diagnosis (CAD) of skin cancers, such as melanoma, accurate segmentation of lesion boundaries is fundamental to enabling reliable quantitative analysis and diagnostic decision-making. To address clinical challenges like diverse lesion morphologies and ambiguous boundaries, we introduce HCViT-Net, a novel hybrid CNN-ViT architecture. This network is designed to synergistically leverage the powerful local detail extraction capabilities of Convolutional Neural Networks (CNNs) and the superior global context modeling strengths of Vision Transformers (ViTs). Conventional hybrid models, constrained by the high computational cost of self-attention mechanisms, typically introduce ViT modules only in deep network stages, which limits the early utilization of global information. To overcome this bottleneck, we design a lightweight Multi-Scale Query Transformer (MSQFormer). Through an innovative multi-scale key-value compression strategy, this module significantly reduces computational complexity while efficiently preserving the core advantages of global context modeling. More importantly, by embedding MSQFormer after each CNN block, we enable full-scale feature co-optimization from local to global contexts, spanning from shallow to deep layers. This design equips the model with high segmentation robustness, particularly when processing challenging lesions with ambiguous boundaries or complex backgrounds. Extensive experiments on the benchmark ISIC 2017 and ISIC 2018 datasets demonstrate that HCViT-Net surpasses leading pure CNN, pure ViT, and other hybrid models in segmentation accuracy. While maintaining a competitive model complexity, our method exhibits immense potential for precise and rapid skin lesion analysis in realworld clinical settings, poised to become an effective tool to assist clinicians in improving both diagnostic efficiency and accuracy

## Data Preprocessing

Please follw the [MALUNet](https://github.com/JCruan519/MALUNet) to prepocess the ISIC 2017 and ISIC 2018 dataset.

## Training

"-c" means the path of the config, use different **config** to train different models.

**train ISIC 2017:**

```shell
CUDA_VISIBLE_DEVICES=0 python train.py -c ./config/isic_2017/hcvitnet.py
```
**train ISIC 2018:**

```shell
CUDA_VISIBLE_DEVICES=0 python train.py -c ./config/isic_2018/hcvitnet.py
```

## Testing

**eval ISIC 2017:** 

```shell
CUDA_VISIBLE_DEVICES=0 python evaluate.py -c ./config/isic_2017/hcvitnet.py -o ./fig_results/isic_2017/hcvitnet/ --rgb 
```

**eval ISIC 2018:** 
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate.py -c ./config/isic_2018/hcvitnet.py -o ./fig_results/isic_2018/hcvitnet/ --rgb 
```
