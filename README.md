# Quick Human Actions Recognition

## Introduction
This repository holds the codebase and dataset for the project:

**Spatial Temporal Graph Convolutional Networks for the Recognition of Quick Human Actions**

## Prerequisites
- Python3 (>3.5)
- [PyTorch](http://pytorch.org/)

## Data Preparation
We Experimented on the 3D Skeletal Data of **NTU-RGB+D**. <br/>
The pre-processed data can be downloaded from
[GoogleDrive](https://drive.google.com/open?id=103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb). <br/>
After downloading the data, extract the "NTU-RGB-D" folder into path.
   
## Downsampling
To create a dataset of fast actions, we downsample the NTU-RGB+D dataset. <br/>
The downsampling is done by taking one frame and leaving another, halving the number of frames. <br>
Run "downsample.py" to downsample the desired data.

## Data Reduction (optional)
We provide "create_small_data.py" that creates a smaller data from the original data by selecting a number of actions out of all 60 actions.
The desired actions can be selected in the code based on their labels on [the NTU-RGB+D website](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).

## Visualization
We provide visualization of the 3D skeletal data of **NTU-RGB+D** on MATLAB. <br/>

![output](https://user-images.githubusercontent.com/68873733/117915304-af1f7600-b2ed-11eb-811f-313261572cff.gif)

More details can be found on the "visualize" folder.

## Results
Some results of different experiments are shown here:

| Model | Temporal Kernel Size | Downsampled NTU-RGB+D <br/> (60 actions)| Downsampled NTU-RGB+D <br/> (10 actions) |
| :------ | :------: | :------: | :------: |
| Model I (ST-GCN) [1] | 9 | 76.29% | 93.39% |  
| **Model II** (Proposed)| **9** | - | **94.01%** | 
| Model I (ST-GCN) [1] | 13 | - | 94% |  
| **Model II** (Proposed)| **13** | - | **93.29%** | 

[1] Sijie Yan et al., 2018. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition.

