## -MARKING LANE LINES AND DETERMINING RADIUS OF CURVATURE OF ROAD-
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







The preliminary task in self driving cars is to detect the lanes.
### Step 1: Distortion correction
The code for this step is contained in camera_caliberation.py file.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output objpoints and imgpoints to compute the camera calibration and distortion coefficients using the cv2.calibrateCamera() function. I applied this distortion correction to the test image using the cv2.undistort() function and obtained this result:

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/cam_caliberation.JPG)
![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/undist.JPG)
![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/chooser.JPG)
![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/perspective.JPG)


### Step 2:  Pipeline 
architecture: The following flow method is used for processing the images. 
![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/architecture.JPG)

### Step 3:
step by step testing of pipeline on a image.
![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/white_filter.JPG)
![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/yellow_filter.JPG)
![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/combined.JPG)
![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/test_on_all.JPG)
We implement different lane calculations for the first frame and subsequent frames. In the first frame, we compute the lanes using computer vision methods, however, in the later frames, we skip these steps. Instead, we place windows of 50 pixel width centered on the lanes computed in the previous frame, and search within these windows. This significanly reduced the computation time, for our algorithm. We were able to achieve 10 Frames/s lane estimation rate.

The next step is to compute lanes for the first image. To do so, we take the lane mask from the previous step, and take only the bottom half of the image. We next use scipy to compute the locations of the peaks corresponding to the left and right lanes.

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/hista.JPG)

We then place a window of size 50 pixels centered at these peaks, and search for peaks in the bottom 8th of the image. Next we move up to the next 1/8th of the image and center windows at the peaks detected in the bottom 1/8th of the image. We repeat this process 8 times to cover the entire image. This is illustrated in the figure below.

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/hist.JPG)

After computing the lanes, we draw them back on the original undistorted image as follows.

![alt text](https://github.com/GOUTHAMRANGU/SDCND-UDACITY/blob/master/PROJECT4/output/test_final.JPG)

### Step 4:
If the current frame is not the first frame, we follow the same steps as part 2 to get the lane masks, however, we introduced additional steps to ensure any error due to incorrectly detected lanes is removed. Lane correction are introduced as,

Outlier removal 1: If the change in coefficient is above 0.008, the lanes are computed as (coeffs = 0.5*coeff~prev+ 0.5 coeff\).. This number was obtained empirically.
Outlier removal 2: If any lane was found with less than 6 pixels, we use the previous line fit coefficients as the coefficients for the current one.
Smoothing: We smooth the value of the current lane using a first order filter response, as \(coeffs = 0.6*coeff~prev+ 0.4 coeff\).
Finally , we use the coefficients of polynomial fit to compute curvatures of the lane, and relative location of the car in the lane.
### Step 5:
Output of pipoeline on given videos:

[result on simple video](https://youtu.be/RLADQ1ScPZk)
[result on challenge 1](https://youtu.be/kAPKyNAQ1QI)

