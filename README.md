# Stacked Hourglass Network with a Multi-level Attention Mechanism: Where to Look for Vertebral Disc Labeling


## Introduction
Automatic labeling of the vertebral disc is a difficult task, due to the many challenges such as complex background, the similarity between discs and bone area in MRI imaging, blurry image, and variation in an imaging modality. Precisely localizing spinal discs plays an important role in vertebral disc labeling. Most of the literature work consider the semantic vertebral disc labeling as a post-processing step, which applies on the top of the disc localization algorithm. Hence, the semantic vertebral labeling highly depends on the disc localization algorithm and mostly fails when the localization algorithm cannot detect discs or falsely detects a background area as a disc. In this work, we aim to mitigate this problem by reformulating the semantic disc labeling using the pose estimation technique. To do so, we propose a stacked hourglass network with a multi-level attention mechanism to estimate the vertebral disc location in the MRI images. The proposed deep model takes into account the strength of semantic segmentation and pose estimation technique to handle the missing and falsely additional disc areas. The skeleton structure of the vertebral discs is shown in figure 1.

![Skeleton](https://github.com/rezazad68/DeepSpine/blob/main/images/skeleton3.png)
##### <pre>                                            Fig. 1. Skeleton structure of the vertebral discs</pre>

## Material and method

### Dataset
In this work, we use the publicly available spinal cord dataset ([Cohen-Adad, 2019](https://github.com/spine-generic/data-multi-subject#spine-generic-public-database-multi-subject)). The dataset contains both MRI T1w and T2W modalities for 251 subjects, which acquired from 40 different centers. The dataset contains high variability in terms of quality, ratio, and structure. We follow the literature work ([Rouhier, 2019](https://arxiv.org/pdf/2003.04387.pdf)) and extract the average of 6 middle slides from each subject data (either T1W or Tw2) as a 2D image for the training purpose. Each image is then pre-processed using sample-wise normalization. Furthermore, We extract a single pixel per vertebral disc as a ground truth mask using the manually annotated data provided by the [Ivadomed library](https://ivadomed.org/en/latest/). 

### Data augmentation
We only use the fliping as a data augmentation method. Since the model requires each vertebral disc location in a different channel, we use the morphological approach to separate the vertebral disc location on the ground truth heatmap. On top of the separated discs we apply the gaussian kerne with sigma n to generate a smooth annotation (similar to [Rouhier, 2019](https://arxiv.org/pdf/2003.04387.pdf)).

### Training process
We train the model for 200 epochs using 80% of the dataset. At the end of each epoch, we evaluate the model on the validation set (10% of the dataset). The training process uses the sum of MSE loss between the ground truth discs heatmap and all intermediate and final prediction results. The training convergence gif is showing in figure 2, which represents the estimated vertebral disc locations at the end of each epoch for the validation set. As it is clear the model is able to successfully recognize the vertebral disc location with precise order at the later epochs. Please read the run demo section to get a guide to run the code and limitation section to find out issues and challenges that we need to overcome.

![learning convergence](https://github.com/rezazad68/DeepSpine/blob/main/images/Learning%20convergence.gif)
##### <pre>                                            Fig. 2. Learning convergence for prediction </pre>

### Baseline model
In this work, we use the stacked hourglass network as a baseline model. The hourglass network (figure 3) is an encoder-decoder network, which uses asymmetric structure on the encoder and decoder section.  Similar to the U-net model it uses the residual information for enhancing the representation. However, on the transition path, it applies a series of convolution layers to compactly represent the information. The stacked hourglass network consists of N times repeating the hourglass network and successively connecting these networks together.

![Diagram of the proposed method](https://github.com/rezazad68/DeepSpine/blob/main/images/stackedhourglass.png)
##### <pre>                                            Fig. 3. Stacked hourglass network </pre>

### Stacked hourglass network with multi-level attention mechanism
The stacked hourglass network learns the object pose using (N-1) intermediate prediction and one final prediction. Thus, it takes into account the multi-level representation in terms of the N stacked hourglass network. To further improve the power of representation space, we propose to use a multi-level attention mechanism. To do so, an intermediate representation generated by each hourglass network are concatenated to form a multi-level representation. In fact, this representation can be seen as a collective knowledge that extracted from a different level of the network with various scale, thus, using this collective knowledge as a supervisory signal to scale the final representation can result in better representation. To do so, we stack all the intermediate representation, then we feed this stacked representation to the attention block (series of convolution with a point-wise kernel) to generate a single channel attention mechanism. We multiply this attention channel with the final representation to help the model to pay more attention to the disc location. See `pose_code/atthourglass.py` for the implementation details. The figure 4 shows the proposed structure. 

![Diagram of the proposed method](https://github.com/rezazad68/DeepSpine/blob/main/images/proposed%20method2.png)
##### <pre>                         Fig. 4. Stacked hourglass network with multi-level attention mechanisim </pre>

## Prerequisties and Run
This code has been implemented in python language using Pytorch libarary and tested in ubuntu, though should be compatible with related environment. The required libraries are included in the requiremetns.txt file. Please follow the bellow steps to train and evaluate the model. </br>

1- Download the [Spine Generic Public Database (Multi-Subject)](https://github.com/spine-generic/data-multi-subject#spine-generic-public-database-multi-subject).</br>
2- Run the `create_dataset.py` to gather the required data from the Spin Generic dataset. </br>
3- Optinonall at the moment (run the `generate_straightened.bash`) to generate the straightened images using the [spinal cord toolbox](https://spinalcordtoolbox.com/en/stable/index.html) library. </br>
4- Run `prepare_trainset.py` to creat the training and validation samples. </br>
Notice: To avoid the above steps we have provided the processed data [here]() you can simply download it and continue with the rest steps. 
5- Run the `main.py` to train and evaluate the model. Use the following command with the related arguments to perform the required action:</br>

A- Train and evaluate the model `python src/main.py`. You can use `--att true` to use the attention mechanisim. </br>
B- Evaluate the model `python src/main.py --evaluate true ` it will load the trained model and evalute it on the validation set. </br>
C- You can run `make_res_gif.py` to creat a prediction video using the prediction images generated by `main.py` for the validation set.  </br>
D- You can change the number of stacked hourglass by `--stacks ` argument. For more details check the arguments section in `main.py`.

#### Visualzie the attention channel

To extract and show the attention channel for the related input sample, we registered the attention channel by the forward hook. Thus with the following command, you can visualize the input sample, estimated vertebral disc location, and the attention channel. </br>
`python src/main.py --evaluate true --attshow true `. </br>

![Attention visualization](https://github.com/rezazad68/DeepSpine/blob/main/images/attention_visualization.png)
##### <pre>                                            Fig. 5. Attention mechanisim visualization </pre>

### Limitaion and issues: 
We srot the spin generic dataset and choose the first 80% as a train set. Figure 6 shows the sample of extracted images for the training purpose. 
![Train data](https://github.com/rezazad68/DeepSpine/blob/main/images/data_train.png)
##### <pre>                                            Fig. 6. Training samples </pre>

10% of the dataset is also considered as a validation set. 
![Train data](https://github.com/rezazad68/DeepSpine/blob/main/images/data_val.png)
##### <pre>                                            Fig. 7. Validation samples </pre>
The rest of the data is considered as test set. As we can see there are large variation between the train/validation and test sets. Thus, the models performance on test set is not good. 
![Train data](https://github.com/rezazad68/DeepSpine/blob/main/images/data_test.png)
##### <pre>                                            Fig. 8. Test samples </pre>

#### Issues to handle
1- How should I devide the dataset into train, validation and test set to handle the above mentioned problem? In fact the data variablity comes from the center variation and training a model with small dataset and different structured data can largely reduce the performance. </br>

2- In the current implementation I have not used the straightening strategy, does straightening can solve the above-mentioned problem? In the case of straightening which data from the figure 9 should be used as data and annotation? </br>

![Train data](https://github.com/rezazad68/DeepSpine/blob/main/images/straigthening.png)
##### <pre>                                            Fig. 9. spine generic files for each subject </pre>

3- What should I add as a data agumentation method? </br>

4- What would you suggest as a additional loss? </br>

### To do
1- Fidx the issues </br>
2- Add the evaluation metrics </br>
3- Compare the resutls with the litreature work </br>
4- Enhance the implementation </br>
5- Read and use the local maxima idea https://github.com/bmartacho/UniPose

