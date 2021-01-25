# Stacked Hourglass Network with a Multi-level Attention Mechanism: Where to Look for Vertebral Disc Labeling


## Introduction
Automatic labeling of the vertebral disc is a difficult task, due to the many challenges such as complex background, the similarity between discs and bone area in MRI imaging, blurry image, and variation in an imaging modality. Precisely localizing spinal discs plays an important role in vertebral disc labeling. Most of the literature work consider the semantic vertebral disc labeling as a post-processing step, which applies on the top of the disc localization algorithm. Hence, the semantic vertebral labeling highly depends on the disc localization algorithm and mostly fails when the localization algorithm cannot detect discs or falsely detects a background area as a disc. In this work, we aim to mitigate this problem by reformulating the semantic disc labeling using the pose estimation technique. To do so, we propose a stacked hourglass network with multi-level attention mechanisim to estimate the vertebral disc location in the MRI images. The proposed deep model takes into account the strength of semantic segmentation and pose estimation technique to handle the missing and falsely additional disc areas. The structure of the proposed method is shows in the below figure. 

![Skeleton](https://github.com/rezazad68/DeepSpine/blob/main/images/skeleton.png | width=100)

## Material and method

### Dataset
In this work we use the publickly availble spinal cord dataset (Cohen-Adad, 2019). The dataset contains both MRI T1w and T2W modalies for 251 subjects, which acuired from 40 different centers. The dataset contains high variablity in term of quality, ratio and structure. We follow the litreture wok and extract the average of 6 middle slide from each subject data (either T1W or Tw2) as a 2D image for the training purpose. Each image are then pre-processed using sample-wise normalization. Furthermore, We extract single pixel per vertebral disc as a ground truth mask using the manually annotated data provided by Ivadomed library. 

### Data augmentation
We only use the fliping as a data augmentation method. Since the model requires each vertebral disc location in different channel, we use the morphological approach to seperate the disc. On top of the seperated disc we apply the guassian kerne with sigma n to generate a smooth annotation (similar to Rouhier, 2019). sample of data along with its annotation is shown in the bellow figure.  

![Diagram of the proposed method](https://github.com/rezazad68/DeepSpine/blob/main/images/proposed%20method.png)

### Training process
We train the model for 200 epochs usign 80% of the dataset. At the end of each eopoch we evaluate the model on the validation set (10% of the dataset). The training convergence gif is showing below, which represent the estimated vertebral disc locations at the end of each eopoch for the validation set. As it is clear the model is able to successfuly recoginze the vertebral disc location with precise order at the later epochs. Please read the run demo section to get a guide to run the code and limitation section to find out issues and challenges that we need to overcome. 


## Prerequisties and Run
This code has been implemented in python language using Pytorch libarary and tested in ubuntu, though should be compatible with related environment. The required libraries are included in the requiremetns.txt file. Please follow the bellow steps to train and evaluate the model. 
1- Download the [Spine Generic Public Database (Multi-Subject)](https://github.com/spine-generic/data-multi-subject#spine-generic-public-database-multi-subject).
2- Run the `create_dataset.py` to gather the required data from the Spin Generic dataset. 
3- Optinonall at the moment (run the `generate_straightened.bash`) to generate the straightened images using the [SCT]() library. 
4- Run `prepare_trainset.py` to creat the training and validation samples. 
Notice: To avoid the above steps we have provided the processed data [here]() you can simply download it and continue with the rest steps. 
5- 

1- Download the ISIC 2018 train dataset from [this](https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53) link and extract both training dataset and ground truth folders inside the `dataset_isic18`. </br>
2- Run `Prepare_ISIC2018.py` for data preperation and dividing data to train,validation and test sets. </br>
3- Run `train_isic18.py` for training BCDU-Net model using trainng and validation sets. The model will be train for 100 epochs and it will save the best weights for the valiation set. You can also train U-net model for this dataset by changing model to unet, however, the performance will be low comparing to BCDU-Net. </br>
4- For performance calculation and producing segmentation result, run `evaluate.py`. It will represent performance measures and will saves related figures and results in `output` folder.</br>

#### Sample resutl

![Skin Lesion Segmentation result 1](https://github.com/rezazad68/LSTM-U-net/blob/master/output_images/1%20(1).png)


## Limitaion and issues: 


## To do




```
