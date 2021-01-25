# Stacked Hourglass Network with a Multi-level Attention Mechanism: Where to Look for Vertebral Disc Labeling


## Overview and objective
Automatic labeling of the vertebral disc is a difficult task, due to the many challenges such as complex background, the similarity between discs and bone area in MRI imaging, blurry image, and variation in an imaging modality. Precisely localizing spinal discs plays an important role in vertebral disc labeling. Most of the literature work consider the semantic vertebral disc labeling as a post-processing step, which applies on the top of the disc localization algorithm. Hence, the semantic vertebral labeling highly depends on the disc localization algorithm and mostly fails when the localization algorithm cannot detect discs or falsely detects a background area as a disc. In this work, we aim to mitigate this problem by reformulating the semantic disc labeling using the pose estimation technique. To do so, we propose a stacked hourglass network with multi-level attention mechanisim to estimate the vertebral disc location in the MRI images. The proposed deep model takes into account the strength of semantic segmentation and pose estimation technique to handle the missing and falsely additional disc areas. The structure of the proposed method is shows in the below figure. 

![Diagram of the proposed method](https://github.com/rezazad68/DeepSpine/blob/main/images/proposed%20method.png)

## Training process
We train the model using spin generic public dataset. 

An onverview of the training convergence on the validation set is shown in the below figure. 


![](https://github.com/neuropoly-mila/deeppose/blob/main/images/Learning%20convergence.gif)



## Updates

- January 25, 2021: Initial implementation completed. 

## Prerequisties and Run
This code has been implemented in python language using Pytorch libarary and tested in ubuntu, though should be compatible with related environment. The required libraries are included in the requiremetns.txt file.

## Run Demo
Please follow the bellow steps to train and evaluate the model. 
1- Download the ISIC 2018 train dataset from [this](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `train_isic18.py` for training BCDU-Net model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. You can also train U-net model for this dataset by changing model to unet, however, the performance will be low comparing to BCDU-Net. </br>
4- For performance calculation and producing segmentation result, run `evaluate.py`. It will represent performance measures and will saves related figures and results in `output` folder.</br>

#### Sample resutl

![Skin Lesion Segmentation result 1](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/1%20(1).png)


## Limitaion and issues: 


## To do




```
