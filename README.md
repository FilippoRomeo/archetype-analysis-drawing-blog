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
**hardware used:**
* Jeston nano
* Arduino Leonardo
* Webcam
* 2x servos 
* 2x solenoid
* laser module

**libraries:**
* Jetson inference with python binding  
* Pyfirmata, standard-firmatata 
* Opencv aruco
 
## Posenet with Jetson 
To extrapolate the pose estimation, I used the jetson nano, excellent hardware that contains 128 Cuda cores, allowing a DNN to run on real-time and images
and produce immediate results. 
Posenet returns two essential data a 2D array called *Links* and an object called *keypoints*. The 2D array stores information about the joins between body
parts ("links") needed to produce the skeletal topology. The keypoints object returns a list of items containing id and coordinates x and y required to
identify the body parts.
- Links
```
 [(12, 14), (10, 12), (13, 15), (11, 13), (10, 11), (5, 7), (6, 8), (7, 9), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (0, 16), (5, 16), (6, 16), (10, 16), (11, 16)]
```
- Keypoints
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
   -- x:   913.639
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

