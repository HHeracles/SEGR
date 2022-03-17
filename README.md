# SEGR:Semantic Enhancement and Graph Reasoning for Scene Text Recognition
The SEGR scene text recognition algorithm proposed in this paper mining semantic information through semantic enhancement module for text recognition based on semantic enhancement, and then correcting the misrecognized text characters according to context information through graph reasoning mechanism, which improves the text recognition accuracy.  
## Requirements
```
pip install torch==1.7.1 torchvision==0.8.2 fastai==1.0.60 opencv-python tensorboardX lmdb pillow
```
## Datasets
We used datasets in LMDB format for training and evaluation. Synthetic datasets MJSynth, SynthTex and WikiText were used in the training process, and three irregular text datasets and three regular text datasets were used in the evaluation process.\<br>
*training datasets\<br>
  *MJSynth\<br>
  *SynthTex\<br>
  *WikiText\<br>
*Evaluation datasets\<br>
The evaluation data set can be downloaded from  BaiduNetdisk(passwd:1dbv or GoogleDrive.  It can also be downloaded from the corresponding official website. \<br>
  *Regular scene text datasets\<br>
    *ICDAR2013(IC13)\<br>
    *Street View Text(SVT)\<br>
    *IIIT5k(IIIT)\<br>
  *Irregular scene text datasets\<br>
    *ICDAR2015 (IC15)\<br>
    *SVT Perspective(SVTP)\<br>
    *CUTE80(CUTE)\<br>
*The directory structure of the dataset is as follows:\<br>
