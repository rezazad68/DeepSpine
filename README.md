# Stacked Hourglass Network with a Multi-level Attention Mechanism: Where to Look for Vertebral Disc Labeling


## Introduction
Automatic labeling of the intervertebral disc is a difficult task, due to the many challenges such as complex background, the similarity between discs and bone area in MRI imaging, blurry image, and variation in an imaging modality. Precisely localizing spinal discs plays an important role in intervertebral disc labeling. Most of the literature work consider the semantic intervertebral disc labeling as a post-processing step, which applies on the top of the disc localization algorithm. Hence, the semantic intervertebral labeling highly depends on the disc localization algorithm and mostly fails when the localization algorithm cannot detect discs or falsely detects a background area as a disc. In this work, we aim to mitigate this problem by reformulating the semantic intervertebral disc labeling using the pose estimation technique. To do so, we propose a stacked hourglass network with a multi-level attention mechanism to estimate the intervertebral disc position in the MRI images. The proposed deep model takes into account the strength of semantic segmentation and pose estimation technique to handle the missing and falsely additional disc areas. The skeleton structure of the intervertebral discs is shown in figure 1.

![Skeleton](https://github.com/rezazad68/DeepSpine/blob/main/images/skeleton3.png)
##### <pre>                                            Fig. 1. Skeleton structure of the intervertebral discs</pre>

### Updates
2-25-2021: code finalized. </br>
1-28-2021: Generalization issue solved, now model works fine on the test set. </br>
1-25-2021: initial version of the implementation is out now


## Material and method

### Dataset
In this work, we use the publicly available spinal cord dataset ([Cohen-Adad, 2019](https://github.com/spine-generic/data-multi-subject#spine-generic-public-database-multi-subject)). The dataset contains both MRI T1w and T2W modalities for 251 subjects, which acquired from 40 different centers. The dataset contains high variability in terms of quality, ratio, and structure. We follow the literature work ([Rouhier, 2019](https://arxiv.org/pdf/2003.04387.pdf)) and extract the average of 6 middle slides from each subject data (either T1W or Tw2) as a 2D image for the training purpose. Each image is then pre-processed using sample-wise normalization. Furthermore, We extract a single pixel per vertebral disc as a ground truth mask using the manually annotated data provided by the [Ivadomed library](https://ivadomed.org/en/latest/). 

### Data augmentation
We only use the fliping as a data augmentation method. Since the model requires each intervertebral disc location in a different channel, we use the morphological approach to separate the vertebral disc location on the ground truth heatmap. On top of the separated discs we apply the gaussian kerne with sigma 10 to generate a smooth annotation (similar to [Rouhier, 2019](https://arxiv.org/pdf/2003.04387.pdf)).

### Training process
We train the model for 200 epochs using 75% of the dataset as a train set. At the end of each epoch, we evaluate the model on the validation set (10% of the dataset). The training process uses the sum of MSE loss between the ground truth discs heatmap and all intermediate and final prediction results. The training convergence gif is showing in figure 2, which represents the estimated intervertebral disc locations at the end of each epoch for the validation set. As it is clear the model is able to successfully recognize the intervertebral disc location with precise order at the later epochs. Please read the run demo section to get a guide to run the code.

![learning convergence](https://github.com/rezazad68/DeepSpine/blob/main/images/Learning%20convergence.gif)
##### <pre>                                            Fig. 2. Learning convergence for prediction </pre>

### Baseline model
In this work, we use the stacked hourglass network as a baseline model. The hourglass network (figure 3) is an encoder-decoder network, which uses asymmetric structure on the encoder and decoder section.  Similar to the U-net model it uses the residual information for enhancing the representation. However, on the transition path, it applies a series of convolution layers to compactly represent the information. The stacked hourglass network consists of N times repeating the hourglass network and successively connecting these networks together.

![Diagram of the proposed method](https://github.com/rezazad68/DeepSpine/blob/main/images/stackedhourglass.png)
##### <pre>                                            Fig. 3. Stacked hourglass network </pre>

### Stacked hourglass network with multi-level attention mechanism
The stacked hourglass network learns the object pose using (N-1) intermediate prediction and one final prediction. Thus, it takes into account the multi-level representation in terms of the N stacked hourglass network. To further improve the power of representation space, we propose to use a multi-level attention mechanism. To do so, an intermediate representation generated by each hourglass network are concatenated to form a multi-level representation. In fact, this representation can be seen as a collective knowledge that extracted from a different level of the network with various scale, thus, using this collective knowledge as a supervisory signal to scale the final representation can result in better representation. To do so, we stack all the intermediate representation, then we feed this stacked representation to the attention block (series of convolution with a point-wise kernel) to generate a single channel attention mechanism. We multiply this attention channel with the final representation to help the model to pay more attention to the intervertebral disc location. See `pose_code/atthourglass.py` for the implementation details. The figure 4 shows the proposed structure. 

![Diagram of the proposed method](https://github.com/rezazad68/DeepSpine/blob/main/images/proposed%20method2.png)
##### <pre>                         Fig. 4. Stacked hourglass network with multi-level attention mechanisim </pre>

### Post-processing
Model prediction often contains noisy area which needs to be eliminated. To do so, we create a skeleton structure from the trainin set then for each new subject we build a search tree based on prediciton masks. In the search tree, the path with minimum distance to the skeleton structure selects as a final prediciton. Figure 5 shows the skeleton structure extracted from the training set.  

![Diagram of the proposed method](https://github.com/rezazad68/DeepSpine/blob/main/images/skeleton_distribution.png)
##### <pre>                         Fig. 5. Skeleton structure and search tree to recover intervertebral disc location on predicted masks </pre>


## Prerequisties and Run
This code has been implemented in python language using Pytorch libarary and tested in ubuntu, though should be compatible with related environment. The required libraries are included in the requiremetns.txt file. Please follow the bellow steps to train and evaluate the model. </br>

1- Download the [Spine Generic Public Database (Multi-Subject)](https://github.com/spine-generic/data-multi-subject#spine-generic-public-database-multi-subject).</br>
2- Run the `create_dataset.py` to gather the required data from the Spin Generic dataset. </br>
4- Run `prepare_trainset.py` to creat the training and validation samples. </br>
Notice: To avoid the above steps we have provided the processed data [here](https://drive.google.com/file/d/1z_mcIEoT_doyh_Hl53OaYWyplUel_-RT/view?usp=sharing) you can simply download it and continue with the rest steps. 
5- Run the `main.py` to train and evaluate the model. Use the following command with the related arguments to perform the required action:</br>
A- Train and evaluate the model `python src/main.py`. You can use `--att true` to use the attention mechanisim. </br>
B- Evaluate the model `python src/main.py --evaluate true ` it will load the trained model and evalute it on the validation set. </br>
C- You can run `make_res_gif.py` to creat a prediction video using the prediction images generated by `main.py` for the validation set.  </br>
D- You can change the number of stacked hourglass by `--stacks ` argument. For more details check the arguments section in `main.py`.
6- Run the `test.py` to evaluate the model on the test set alongside with the metrics.  
#### Visualzie the attention channel

To extract and show the attention channel for the related input sample, we registered the attention channel by the forward hook. Thus with the following command, you can visualize the input sample, estimated vertebral disc location, and the attention channel. </br>
`python src/main.py --evaluate true --attshow true `. </br>

![Attention visualization](https://github.com/rezazad68/DeepSpine/blob/main/images/attention_visualizationnew.png)
##### <pre>                                            Fig. 6. Attention mechanisim visualization </pre>


#### Sample of detection result on the test set
Below we illustrated a sample of vertebral disc detection on the test set. 

![Test sample](https://github.com/rezazad68/DeepSpine/blob/main/images/test_result1.png)
##### <pre>                                            Fig. 7. Sample of test results </pre>

### Model weights
You can download the learned weights for each modality in the following table. 

Method | Modality |Learned weights
------------ | -------------|----
Proposed model without attention | T1w | [download](https://drive.google.com/file/d/102U8NlSIelkEmSu4J-Qw-djKsKi6dudt/view?usp=sharing)
Proposed model without attention | T2w | [download](https://drive.google.com/file/d/1pzGDRwFSWb6FN3o8GZD2xrH3gNcPjImt/view?usp=sharing)
Proposed model with    attention | T1w | [download](https://drive.google.com/file/d/1o5DWzHlhMDic5eynrEQMupjZwsmCr5XP/view?usp=sharin)
Proposed model with    attention | T2w | [download](https://drive.google.com/file/d/1zvBbiCVH1gnrYUbzVF6JAUpcpQg7I2dn/view?usp=sharing)

