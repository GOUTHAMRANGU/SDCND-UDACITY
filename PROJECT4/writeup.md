Lot of efort has been put on this project as this is the priliminary task of self driving vehicles
## --------MARKING LANE LINES AND DETERMINING RADIUS OF CURVATURE OF ROAD ------##
#### INTRODUCTION:
Lane detection and radius of curvature determination are very much important for any high level algorithm to predict the required steering angle and accelaration in sel-driven cars. So, algorithms developed to identify lane lines and curvature along with the position of the vehicle in the lane should be fast and accurate enough to be observed in real-time. In this project I try to present a novel idea based on advanced computer vision techniques that stand still in most of the conditions. The important aspects that any algorithm should look into while solving for this problem are shadows and sudden changes in sun light intensitties. So, we convert our image space from RGB to HSV or HSL and create thresholds on saturation values for Region of interest.

#### METHODOLOGY:
This problem is split into two sub-problems. The first sub-problem deals with identifying potential regions inthe image that can contribute to be part of a lane. This is achieved by applying cv techniques that shall be discussed clearly in upcoming steps. The second sub-problem deals with marking lane lines using windows on the original frame to compute radius of curvature and removal of noise and disttubances in the lane lines obtianed by taking into consideration of previous frame lane data.

#### Problem 1: Creating a mask that can detect lanes
The computer vision techniques employed in creating a mask are undistorting, perspective transformation, color masking and sobel filter. All these techniques are used as functions from OpenCV libraries. The design heirarchy is divided into following steps.

1. Read an image from the dataset and undistort using pre-computed distortion camera matrix.
2. Apply perspective transform on undistorted image to have a bird's eye view of the road. Warping the image to right region of interest does good help in further processing of the frame.
3. Shifting the colour space from RGB to either HSV or HSL helps to figure out more data with less effort. HSV values for Yellow range from [0,100,100] to [80,255,255], and for white colour they range from [0,0,210] to [200,50,255]. 'H' or HUE stands for the colour itself , 'S' stands for SATURATION or the amount of colour, and 'V' stands for VALUE or the amount of brighness applied over that colour. As the important aspect of this problem is to find lanes even with heavy variations of brightness, shifting the colour space from RGB to HSV will do a good job. 
4. Once the color mask is applied on the image the next step is to detect the edges from the masked image. Sobel filter is applied on H, L channels of the warped image as these channels show very mimimal change with lighting variations. Sobel filter is applied along X and Y directons independently and a bitwise OR operation is performed on the the masks to obtain a final mask. After many number of iterations the threshold values from 50 to 225 were choosen to be appropriate.
5. For the final step of stage 1 pixels from Colour and Sobel masks are combined to obtain potential lane locations.

#### Problem 2: Computing Curvature
The above methodologies work only in the conditions when the lanes are visible to a good extent, so we need some better way that can propogate the lane forward when there are small disturbances covering parts of the road, like snow, dust, wear and tear. For this reason we run a small window of 50 pixels along the center of the lanes computed in the previous frames. This windowing technique applied from the second frame can help reduce the computation time to a great extent and also helps in continuous propogation of lane. Two correction steps are also followed along with the previous steps to keep the lane smooth, they are outlier removal and smoothing. Pixels are marked as outliers if their width is found to be less than 5 pixles and if coffecient change is more than 0.01 is introduced because of it. A factor of 0.01 is taken keeping vehicle turn radius as atleast 7m corresponding to short side lane  curvature. Smoothing is achieved by adopting 90 percent of previous coefficients into the present quadratic coefficient values. Using these coefficients of new polynomial fit we compute the curvature of the lane and relative location of the car in the lane. Followings steps are the pipeline for this stage.

1. Consider only half of the image that is masked and compute histogram, and we find the lane locations at the two peaks of histogram corresponding to left and right lanes of the road in the masked image.
2. The next step is placing a window of size 50x50 pixels on the bottom 10th part of the image by centering the windows the peaks located in the previous step and propagate forward in the same way to next sections of the image and to upcoming frames. In case if no peaks were found in the previous link we place the window at a location with offset from previous computed center.
3. Now we try fit a quadratic fit polynomial using numpys polyfit function.
4. If the current frame is not the first frame we follow the pervious steps for computation and the next step is outlier removal. If a lane was found with less than 5 pixels it is discarded and the previous coefficients are used in the place of current.
5. However smoothing is required on the lanes, so previous curve coefficients are used along the current values. This helps smooth shifting of lane curvature.
6. Finally the lane curvature and position of car on the lane are computed and reverse marked on the unwarped perspective image.



[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
