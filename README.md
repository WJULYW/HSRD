# HSRD (Hierarchical Style-aware Representation Disentangling)
Official code of IEEE JBHI 2024 ["Hierarchical Style-Aware Domain Generalization for Remote Physiological Measurement"](https://ieeexplore.ieee.org/abstract/document/10371379).

<div style="text-align:center;">
  <img src="Framework.png" style="width:100%;" />
</div>

## Data Prepare
You can refer to [link](https://github.com/EnVision-Research/NEST-rPPG) to obtain the processed STMaps.
Before that, please get the permission to use the following datasets first:
[**VIPL**](http://vipl.ict.ac.cn/en/resources/databases/201901/t20190104_34800.html),
[**V4V**](https://competitions.codalab.org/competitions/31978),
[**BUAA**](https://ieeexplore.ieee.org/document/9320298),
[**UBFC**](https://sites.google.com/view/ybenezeth/ubfcrppg), 
[**PURE**](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/pulse-rate-detection-dataset-pure).
After getting STMaps, you can create a new './STMap' folder and put them into it.
For the first time running, please adjust the hyperparameter 'reData' to 1, to generate the STMap index.

## Pre-trained Model
In this work, we utilized the ResNet18 as the backbone network. You can download it directly from this [link](https://download.pytorch.org/models/resnet18-5c106cde.pth).
Next, create a new folder './pre_encoder' and put the pth file into it.

## Train and Test
Then, you can try to train it with the following command:

```
python train.py -g $GPU id$ -t 'the target dataset you want to test on'
```
## Citation
```
@ARTICLE{10371379,
  author={Wang, Jiyao and Lu, Hao and Wang, Ange and Chen, Yingcong and He, Dengbo},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Hierarchical Style-Aware Domain Generalization for Remote Physiological Measurement}, 
  year={2024},
  volume={28},
  number={3},
  pages={1635-1643},
  keywords={Feature extraction;Videos;Skin;Physiology;Biomedical measurement;Bioinformatics;Training;Adversarial learning;contrastive learning;domain generalization;heart rate estimation;remote photoplethysmography (rPPG)},
  doi={10.1109/JBHI.2023.3346057}}

```


