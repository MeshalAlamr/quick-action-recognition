# simple-action-recognition
Goal: Classify actions A2, A10 and A26 of the NTU RGB+D dataset using graph convolutional neural networks.

# Steps
1) Download NTU-RGB-D from:

   https://drive.google.com/file/d/103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb/view
   
   Open the downloaded file -> data -> Extract the "NTU-RGB-D" folder into path.
   
2) Run categorize_data.py for the desired categorization.

### OR: 

   For xview only (no xsub), download the following:

   https://drive.google.com/file/d/1yVB_o05xXAyjgGR0vBea2HGo3cG1lwvH/view

   Extract the "xview" folder into path.

   
# Background
   The size of the data (npy file) describes the following ([Index] = Description):
   - [1] = Number of videos.
   - [2] = x,y,z coordinates.
   - [3] = Number of frames (300 frames = 10s) -- filled with zeros if less than 300.
   - [4] = 25 joints --> 3 coordinates.
   - [5] = Number of people (if one person than the rest are zeros).

# TODO

   What we need to do:
   
   - ~~For each one in the 1st dimension of the labels (pkl file), look for 3, 11 and 27 in the 2nd dimension (they start from 0).~~
   - ~~After that, extract their respective data from the npy file.~~
   - ~~Categorize them into files (each action in a folder).~~
   - Train a model to recognize actions A2, A10 and A26.
