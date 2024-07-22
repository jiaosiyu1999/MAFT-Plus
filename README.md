# MAFT-Plus: Collaborative Vision-Text Representation Optimizing for Open-Vocabulary Segmentation
This is the official implementation of our conference paper : Collaborative Vision-Text Representation Optimizing for Open-Vocabulary Segmentation (**ECCV 2024**).

<div align="center">
<img src="resources/vis1.gif" width="48%">
<img src="resources/vis2.gif" width="48%">
</div>

## Introduction

Pre-trained vision-language models, *e.g.* CLIP, have been increasingly used to address the challenging Open-Vocabulary Segmentation (OVS) task, benefiting from their well-aligned vision-text embedding space. Typical solutions involve either freezing CLIP during training to unilaterally maintain its zero-shot capability, or fine-tuning CLIP vision encoder to achieve perceptual sensitivity to local regions.  However, few of them incorporate vision-text collaborative optimization. Based on this, we propose the Content-Dependent Transfer to adaptively enhance each text embedding by interacting with the input image, which presents a parameter-efficient way to optimize the text representation. Besides, we additionally introduce a Representation Compensation strategy, reviewing the original CLIP-V representation as compensation to maintain the zero-shot capability of CLIP. In this way, the vision and text representation of CLIP are optimized collaboratively, enhancing the alignment of the vision-text feature space. To the best of our knowledge, we are the first to establish the collaborative vision-text optimizing mechanism within the OVS field. Extensive experiments demonstrate our method achieves superior performance on popular OVS benchmarks. In open-vocabulary semantic segmentation, our method outperforms the previous state-of-the-art approaches by +0.5, +2.3, +3.4, +0.4 and +1.1 mIoU, respectively on A-847, A-150, PC-459, PC-59 and PAS-20. Furthermore, in a panoptic setting on ADE20K, we achieve the performance of 27.1 PQ, 73.5 SQ, and 32.9 RQ.

![](resources/framework_6x.png)

### Installation
1. Clone the repository
    ```
    git clone https://github.com/jiaosiyu1999/MAFT_Plus.git
    ```
2. Navigate to the project directory
    ```
    cd MAFT_Plus
    ```
3. Install the dependencies
    ```
    bash install.sh
    cd maft/modeling/pixel_decoder/ops
    sh make.sh
    ```
    
<span id="2"></span>

### Data Preparation
See [MAFT](https://github.com/jiaosiyu1999/MAFT/tree/master) for reference ([Preparing Datasets for MAFT](https://github.com/jiaosiyu1999/MAFT/tree/master/datasets#readme)). The data should be organized like:
```
datasets/
  ade/
      ADEChallengeData2016/
        images/
        annotations_detectron2/
      ADE20K_2021_17_01/
        images/
        annotations_detectron2/
  coco/
        train2017/
        val2017/
        stuffthingmaps_detectron2/
  VOCdevkit/
     VOC2012/
        images_detectron2/
        annotations_ovs/      
    VOC2010/
        images/
        annotations_detectron2_ovs/
            pc59_val/
            pc459_val/      
```
<span id="3"></span>

### Usage


- #### Pretrained Weights

  |Model|A-847| A-150| PC-459| PC-59| PAS-20 |Weights|
  |-----|--|-|-|-|--|---|
  |MAFTP-Base|13.2|33.6|14.2|55.9|93.9 | |
  |MAFTP-Large|15.1|36.1|21.6|59.4|96.5 |[maftp_l.pth](https://drive.google.com/file/d/1EQo5guVuKkSSZj4bv0FQN_4X9h_Rwfe5/view?usp=sharing) |
  

- #### Evaluation 

  <span id="4"></span>
  - evaluate trained model on validation sets of all datasets.
  ```
  python train_net.py --eval-only --config-file <CONFIG_FILE> --num-gpus <NUM_GPU> OUTPUT_DIR <OUTPUT_PATH> MODEL.WEIGHTS <TRAINED_MODEL_PATH>
  ```
   For example, evaluate our pre-trained model:
  ```
  # 1. Download MAFTP-Large.
  # 2. put it at `out/semantic/MAFT_Plus/maftp_l.pth`.
  # 3. evaluation
    python train_net.py --config-file configs/semantic/eval.yaml  --num-gpus 8 --eval-only \
                         MODEL.WEIGHTS out/semantic/MAFT_Plus/maftp_l.pth 
  ```
<span id="5"></span>
- #### Training
1.  **end to end training** requires 8*A100 GPUs and 12 hours, approximately:
```
    python train_net.py --config-file configs/semantic/train_semantic_large.yaml  --num-gpus 8
```

<span id="6"></span>



### Acknowledgement
[Mask2Former](https://github.com/facebookresearch/Mask2Former)

[FC-CLIP](https://github.com/bytedance/fc-clip)




