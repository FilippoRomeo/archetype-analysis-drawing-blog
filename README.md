# archetype analysis drawing blablablog

## Introduction
*Archetype analysis* refers to the idea of extrapolating the body's estimation poses from pictures and reproducing the output on a canvas. 
Archetype refers to Plato's ideas of pure mental forms imprinted in essence and encoded in a newborn individual. Later, Carl Jung used the term in
psychoanalysis, following the concept of undefined preforms that organise a structure that results in being intuitive in mental images. 
Can an AI experience and create similar output?
The main object of this project is to extrapolate, thanks to an AI, the archetype (pose) of dead bodies left from the war in Ukraine. Many websites allow
family members and friends to search, find and retrieve the dead body throughout those websites. Unfortunately, some are hardly identifiable as bodies due
to the brutality suffered. 

## Tech and developement used
**hardware:**
* Jeston nano
* Arduino Leonardo
* Webcam
* 2x servos 
* 2x solenoid
* laser module

**libraries:**
* Jetson inference with python binding  
* Pyfirmata, standard-firmatata 
* Opencv, Aruco, Numpy
 
### Posenet with Jetson 
To extrapolate the pose estimation, I used the jetson nano, excellent hardware that contains 128 Cuda cores, allowing a DNN to run on real-time and images and produce immediate results. I am not going throughout the entire installation of jetkit, tensorrt, jetson utils, jetson inference. I did not use docker I prefered to run things locally for developemnt and debugging reasons.    
The file *bodysk.py* using Posenet uses two essential data which are a 2D array called *Links* and an object called *keypoints*. The 2D array stores information about the joins between body
parts ("links") needed to produce the skeletal topology. The keypoints object returns a list of items containing id and coordinates x and y required to
identify the body parts.
- Links example.
```
 [(12, 14), (10, 12), (13, 15), (11, 13), (10, 11), (5, 7), (6, 8), (7, 9), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 16), (5, 16), (6, 16), (10, 16), (11, 16)]
```
- Keypoints example.
```
[<poseNet.ObjectPose.Keypoint object>
   -- ID:  0 (nose)
   -- x:   844.274
   -- y:   180.49
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  1 (left_eye)
   -- x:   854.904
   -- y:   168.532
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  2 (right_eye)
   -- x:   831.857
   -- y:   168.886
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  3 (left_ear)
   -- x:   870.24
   -- y:   175.787
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  4 (right_ear)
   -- x:   814.22
   -- y:   175.213
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  5 (left_shoulder)
   -- x:   893.444
   -- y:   251.109
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  6 (right_shoulder)
   -- x:   782.821
   -- y:   249.736
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  7 (left_elbow)
   -- x: ###  913.639
   -- y:   341.347
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  8 (right_elbow)
   -- x:   751.509
   -- y:   341.599
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  9 (left_wrist)
   -- x:   923.472
   -- y:   412.009
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  11 (left_hip)
   -- x:   871.635
   -- y:   418.819
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  12 (right_hip)
   -- x:   802.25
   -- y:   418.716
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  13 (left_knee)
   -- x:   869.181
   -- y:   552.813
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  14 (right_knee)
   -- x:   811.697
   -- y:   551.81
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  15 (left_ankle)
   -- x:   860.903
   -- y:   677.762
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  16 (right_ankle)
   -- x:   815.26
   -- y:   677.475
, <poseNet.ObjectPose.Keypoint object>
   -- ID:  17 (neck)
   -- x:   837.951
   -- y:   250.148
]
```
Which if we submit the following image. 
![people](https://user-images.githubusercontent.com/43594252/168493114-bbfa7405-9f10-434f-aa00-e962a87824cc.jpg)
Will return.
![peoplesk](https://user-images.githubusercontent.com/43594252/168493115-332d49c4-37aa-4825-a100-0e3c8bb1d9ad.jpg)

Some adjustments are required to visualize only the skeleton and sort the link connection. I did not use the drawing and visualization method suggested in the documetntation instead I used the cv2 library.

![cv2points](https://user-images.githubusercontent.com/43594252/168493245-95f784be-c7ed-4f66-a640-c63eb346ed91.png)
In order to represent the same points in a new reference system I had to normalized each x and y between 0 and 1, which is important to remember that are in pixels

```
num_rows, num_cols = img.shape[:2]
x = x/(num_cols - 1.)
y = y/(num_rows - 1.)
```
Returning all the normalized points.  

```
[[[0.4544294400182521, 0.10802826838116084], [0.4652589441860643, 0.09470838455219986], [0.44184697991125976, 0.09401847354478254], [0.4784184639521475, 0.1113126386155105], [0.42270482157448985, 0.10977393520172159], [0.5101597581790689, 0.21719483055184013], [0.39370939342390643, 0.21645334444157344], [0.3703281257159549, 0.34746637604116004], [0.3791454386035252, 0.5089704263689607], [0.4983802415059109, 0.4865878212776629], [0.4191061040406586, 0.4890729258496474], [0.48898375092596835, 0.6904554156787046], [0.41117320428962, 0.6855686449973655], [0.48969027909831686, 0.8806483247709954], [0.4096722346480175, 0.8725103831012889], [0.45146556567120294, 0.216198952126905], [0.6304876750748412, 0.1994776546413951], [0.6406976694235704, 0.1861724655606558], [0.6199918357274865, 0.18561325370105788], [0.6525936779272056, 0.201758477474465], [0.6013198509477334, 0.20041966221830415], [0.6683290543094758, 0.29699362669473495], [0.5645238852104717, 0.3113999818240313], [0.6545654774178274, 0.49620430200480303], [0.5898070573107588, 0.49659117464889346], [0.6592482998457356, 0.6834740545963035], [0.5840631897032319, 0.6828527784533074], [0.6408640096147972, 0.8299668209407426], [0.5829181988101784, 0.8173392725052692], [0.6197806290284397, 0.30050890733295843], [0.8252923243905792, 0.23409826375167195], [0.835683753646597, 0.2185883280832016], [0.8131540844750428, 0.21904862499113367], [0.8506741192799976, 0.2279991219170517], [0.7959142481709739, 0.22725381616179272], [0.8733570624656342, 0.3256920048853457], [0.7652207837892534, 0.3239122878098148], [0.893097550288673, 0.44273291363944955], [0.7346131146711571, 0.4430594635999311], [0.902709244982588, 0.5343829401117663], [0.8520378940615836, 0.543215423861057], [0.7842126810888624, 0.5430821914771806], [0.8496391467917583, 0.7170072127563635], [0.7934480818135997, 0.7157067918591926], [0.8415470663986131, 0.8790693270711738], [0.7969306654944098, 0.8786970205344115], [0.8191112469834433, 0.32444646417244244]]]
```
### Visualizing the results
At this point I did not automated and improved the *bodysk.py* file just to develop quicker the outcome and then come back to fix small bugs and especially optimize the overall code. 
So printing the values I copied and paste into a brand new file called *drawing.py* where ideally was designed to identify the canvas and calculate the robot movements. This was the first program that I wrote in order to map the normalized points to the selected area and extract the values for the pan and tilt (the two servos). These are the results:
![1](https://user-images.githubusercontent.com/43594252/168495220-4bdc8d39-4aae-4753-b47b-2a5d2b3d326b.png)
![2](https://user-images.githubusercontent.com/43594252/168495223-c5d4e3ae-a207-4cc2-89d1-0f86569a9ee7.png)
In these two example I am pointing to the wall the camera and testing the normalized points are mapped inside the drawed square. 
The code runs well, in terms of mapping the x ys but if i try to calculate the angle for each servos the returned values are wrong. My approach is to identify the centre of the drawed square, add a distance z, and imagine a traingle rectangle, with the square angle in the center, one angle towards the pan/tilt and another angle to the (x,0) then to and (0,y) so to find the angle for the pan and tilt. 
