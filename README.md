# simple-action-recognition
Goal: Classify actions A2, A10 and A26 of the NTU RGB + D dataset using graph convolutional neural networks

# Steps
1) Download NTU-RGB-D from:
   https://drive.google.com/file/d/103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb/view
   Open the downloaded file -> data -> Extract the "NTU-RGB-D" folder into path
   
# TODO
   The size of the data describes the following ([Index] = Description):
   - [1] = Number of videos
   - [2] = x,y,z coordinates 
   - [3] = 25 joints --> 3 coordinates
   - [4] = Number of frames (300 frames = 10s) -- filled with zeros if less than 300
   - [5] = Number of people (if one person than the rest are zeros)
   
   What we need to do:
   
   - For each one in the 1st dimension of the labels (pkl file), look for 3, 11 and 27 in the 2nd dimension (they start from 0).
   - After that, extract their respective data from the npy file.
   - Categorize them into files (each action in a folder).
   - Train a model to recognize actions A2, A10 and A26.
