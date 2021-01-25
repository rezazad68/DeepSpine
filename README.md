# Stacked Hourglass Network with a Multi-level Attention Mechanism: Where to Look for Vertebral Disc Labeling


## Overview and objective
Automatic labeling of the vertebral disc is a difficult task, due to the many challenges such as complex background, the similarity between discs and bone area in MRI imaging, blurry image, and variation in an imaging modality. Precisely localizing spinal discs plays an important role in vertebral disc labeling. Most of the literature work consider the semantic vertebral disc labeling as a post-processing step, which applies on the top of the disc localization algorithm. Hence, the semantic vertebral labeling highly depends on the disc localization algorithm and mostly fails when the localization algorithm cannot detect discs or falsely detects a background area as a disc. In this work, we aim to mitigate this problem by reformulating the semantic disc labeling using the pose estimation technique. To do so, we propose a stacked hourglass network with multi-level attention mechanisim to estimate the vertebral disc location in the MRI images. The proposed deep model takes into account the strength of semantic segmentation and pose estimation technique to handle the missing and falsely additional disc areas. The structure of the proposed method is shows in the below figure. 

![Diagram of the proposed method](https://github.com/rezazad68/DeepSpine/blob/main/images/proposed%20method.png)



Deep auto-encoder-decoder network for medical image segmentation with state of the art results on skin lesion segmentation, lung segmentation, and retinal blood vessel segmentation. This method applies bidirectional convolutional LSTM layers in U-net structure to non-linearly encode both semantic and high-resolution information with non-linearly technique. Furthermore, it applies densely connected convolution layers to include collective knowledge in representation and boost convergence rate with batch normalization layers. 


![](https://github.com/neuropoly-mila/deeppose/blob/main/images/Learning%20convergence.gif)



## Updates

- January 25, 2021: Initial implementation completed. 

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Keras - tensorflow backend


## Run Demo
For training deep model for each task, go to the related folder and follow the bellow steps:

#### Skin Lesion Segmentation
1- Download the ISIC 2018 train dataset from [this](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `train_isic18.py` for training BCDU-Net model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. You can also train U-net model for this dataset by changing model to unet, however, the performance will be low comparing to BCDU-Net. </br>
4- For performance calculation and producing segmentation result, run `evaluate.py`. It will represent performance measures and will saves related figures and results in `output` folder.</br>

#### Retina Blood Vessel Segmentation
1- Download Drive dataset from [this](https://drive.google.com/open?id=17wVfELqgwbp4Q02GD247jJyjq6lwB0l6) link and extract both training  and test  folders in a folder name DRIVE (make a new folder with name DRIVE) </br>
2- Run `prepare_datasets_DRIVE.py` for reading whole data. This code will read all the train and test samples and will saves them as a hdf5 file in the `DRIVE_datasets_training_testing` folder. </br>
3- The next step is to extract random patches from the training set to train the model, to do so, Run `save_patch.py`, it will extract random patches with size 64*64 and will save them as numpy file. This code will use `help_functions.py`, `spre_processing.py` and `extract_patches.py` functions for data normalization and patch extraction.  
4- For model training, run `train_retina.py`, it will load the training data and will use 20% of training samples as a validation set. The model will be train for 50 epochs and it will save the best weights for the valiation set.</br>
4- For performance calculation and producing segmentation result, run `evaluate.py`. It will represent performance measures and will saves related figures and results in `test` folder.</br>
Note: For image pre-processing and patch extraction we used [this](https://github.com/orobix/retina-unet) github's code.</br>

#### Lung Segmentation
1- Download the Lung Segmentation dataset from [Kaggle](https://www.kaggle.com/kmader/finding-lungs-in-ct-data/data) link and extract it. </br>
2- Run `Prepare_data.py` for data preperation, train/test seperation and generating new masks around the lung tissues.
3- Run `train_lung.py` for training BCDU-Net model using trainng and validation sets (20 percent of the training set). The model will be train for 50 epochs and it will save the best weights for the valiation set. You can train either BCDU-net model with 1 or 3 densly connected convolutions. </br>
4- For performance calculation and producing segmentation result, run `evaluate_performance.py`. It will represent performance measures and will saves related figures and results.</br>



## Quick Overview
![Diagram of the proposed method](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/bcdunet.png)

### Structure of the Bidirection Convolutional LSTM that used in our network
![Diagram of the proposed method](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/convlstm.png)

### Structure of the BConvLSTM+SE that used in our network (MCGU-Net)
![Feature encoder of the MCGU-Net](https://github.com/rezazad68/BCDU-Net/blob/master/output_images/SEBConvLSTM.png)

## Results
For evaluating the performance of the proposed method, Two challenging task in medical image segmentaion has been considered. In bellow, results of the proposed approach illustrated.
</br>
#### Task 1: Retinal Blood Vessel Segmentation


## Skin Lesion Segmentation

#### Performance Evalution on the Skin Lesion Segmentation task

Methods | Year |F1-scores | Sensivity| Specificaty| Accuracy | PC | JS 
------------ | -------------|----|-----------------|----|---- |---- |---- 
Ronneberger and etc. all [U-net](https://arxiv.org/abs/1505.04597)	     	    |2015   | 0.647	|0.708	  |0.964	  |0.890  |0.779 |0.549
Alom  et. all [Recurrent Residual U-net](https://arxiv.org/abs/1802.06955)	|2018	  | 0.679 |0.792 |0.928 |0.880	  |0.741	  |0.581
Oktay  et. all [Attention U-net](https://arxiv.org/abs/1804.03999)	|2018	  | 0.665	|0.717	  |0.967	  |0.897	  |0.787 | 0.566 
Alom  et. all [R2U-Net](https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf)	        |2018	  | 0.691	|0.726	  |0.971	  |0.904	  |0.822 | 0.592
Azad et. all [Proposed BCDU-Net](https://github.com/rezazad68/LSTM-U-net/edit/master/README.md)	  |2019 	| **0.847**	|**0.783**	  |**0.980**	  |**0.936**	  |**0.922**| **0.936**
Azad et. all [MCGU-Net](https://128.84.21.199/pdf/2003.05056.pdf)	  |2020	| **0.895**	|**0.848**	  |**0.986**	  |**0.955**	  |**0.947**| **0.955**




#### Skin Lesion Segmentation results

![Skin Lesion Segmentation result 1](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/1%20(1).png)
![Skin Lesion Segmentation result 1](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/1%20(2).png)
![Skin Lesion Segmentation result 1](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/1%20(3).png)
![Skin Lesion Segmentation result 1](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/1%20(4).png)




```
