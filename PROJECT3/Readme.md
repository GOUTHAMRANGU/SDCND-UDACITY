## IMITATING HUMAN DRIVING BEHAVIOUR USING DEEP NEURAL NETWORKS AND IMPROVISING WITH DATA AUGMENTATION AND FILTERING



#### The main Objective of this project is to build an network that can analyse human driving pattern from a simple track and mimic the same in generalised situations. The idea was to get training data from track1, which is simple to drive, augment it with various types of filtering and image processing techniques to create a datgaset with high affinity to generalize and feed it to a neural network with steering angles as the labels for each image captured during recording. This is one of the best projects that i have made till date.

##### My point about self driving cars is that there is a high probability that a car can drive with only cameras mounted on it for real-time perception using probabilistic modelling, deep-neural networks, and lots of training.

#### In this project the network doesnot consider or remember the previous output that is the steering angle, but by using RNNs we can build a model with greater accuracy to drive on a lane but not at the center of the road as they do have memory.



I used keyboard and mouse as my main means of movement. 

One thing I was quite sure of, while doing this project is that validating the network is not a necessary task as the validation sets with comparibly higher loss were sometimes able to perform well than the others. This leads me to test all the models generated to test. The decision criteria was, if the model was not able to complete atleast one full lap in track1 with out any jerks or running away from tracks that model is considered to be part of result models set. 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


### Python code for this project is broken to parts as listed below
1. Data acquisition and visualization from storage space 
2. Augmenting image data
  * Applying CLAHE to the image data set to equalize the contrast.
  * Cropping the image and rescaling it to the required proportional size.
  * Randomly applying shadows on a portion of the image.
  * Randomly shift the image up and down.
  * Including left and right camera images to the train data set by adding offset to the steering angle and performing the above     techinques on those images.
  * Applying gamma correction to the images.
  * Shearing and rotating a sub-sample of data.
  * Horizontally flipping images and negating the steering angles.
  * Randomly subsampling over the train data.
3. Creating a neural network model that optimises over the train data set collected.
4. Creating an object that keeps track of weights at the end of each epoch and training the network.
5. Choosing the right model weights by testing it on track1.
6. Finally testing the choosen model to work on a generalised track that is Track2.

## Augmentation
step:1 Visualizing the data set

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT3/images/sample.JPG)
----------------------------------------------------------------------------------------------------------------------------------------
step:2 Applying clahe

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT3/images/clahe.JPG)

----------------------------------------------------------------------------------------------------------------------------------------
step:3 Cropping and rescaling the image

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT3/images/crop_reshape_image.JPG)

----------------------------------------------------------------------------------------------------------------------------------------
step:4 Randomly applying shadows

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT3/images/shadow.JPG)

----------------------------------------------------------------------------------------------------------------------------------------
step:5 Applying gamma correction

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT3/images/gamm_diff.JPG)

----------------------------------------------------------------------------------------------------------------------------------------
step:6 Shearing the image

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT3/images/shear.JPG)

----------------------------------------------------------------------------------------------------------------------------------------
step:7 Flipping image

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT3/images/flip.JPG)

----------------------------------------------------------------------------------------------------------------------------------------


# MODEL
![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT3/images/model.JPG)

----------------------------------------------------------------------------------------------------------------------------------------

# RESULT
The results can be viewed on youtube by following the links given below

[Testing the model on track one. The model performed quite well on this track without going off from the center of the track not atleast once. But the actual intention was to test the model on track 2.](https://www.youtube.com/watch?v=-XnEOL9RJ2o)

[Testing model on track two with best graphics and some pre-processing techinques removed, they are shearing and gamma correction. The model tried to perform quite well but got bumped into a side post as there was an overlapping of two tracks in 2d camera view .](https://www.youtube.com/watch?v=tytpl-51GBI)

[This is the final outcome of the model on track two with all possible pre-processing techniques applied on data collected from track one. The model did really well after using shearing and gamma correction. This project can be considered as one of the best projects I have done so far but also the wierdest one. Wierd because of the validation doesnot actually reflect the test performance.](https://www.youtube.com/watch?v=dTX2HpVYWNo)
