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
At this point, I did not automate and improve the *bodysk.py* file just to develop the outcome quicker and then come back to fix minor bugs and optimize the overall code. 
So I printed the values on the console and copied and pasted them into a new file called *drawing.py*, which is ideally designed to identify the canvas and calculate the robot movements. This was the first program that I wrote to map the normalized points to the selected area and extract the values for the pan and tilt (the two servos). These are the results:

![1](https://user-images.githubusercontent.com/43594252/168495220-4bdc8d39-4aae-4753-b47b-2a5d2b3d326b.png)

![2](https://user-images.githubusercontent.com/43594252/168495223-c5d4e3ae-a207-4cc2-89d1-0f86569a9ee7.png)

In these two example I am pointing to the wall the camera and testing the normalized points are mapped inside the drawed square. 

The code runs well in terms of mapping the x ys. But if I try to calculate the angle for each servo, the returned values are wrong. My intent to use trigonometry for each angle failed. First, I find the centre by dividing the width and height by two. The drawn square height adds a distance z, and imagine a triangle rectangle, with the square angle laying on the centre, one angle towards the pan (x,0) and another angle to the tilt (0,y) to find the corrective angle. 

<img width="591" alt="draw" src="https://user-images.githubusercontent.com/43594252/168498338-872cf9ab-09cb-4779-8528-2cc4873a5938.png">

The results are:

* pan:
```
[-7.550697342214455, -8.84811785372994, -8.493479476310146, -15.02354382253232, -15.14680012479041, -21.422908506363378, -20.471515208489368, -3.0832843555430696, 0.8491388992650896, 3.1413629053369863, 4.25873187173949, 4.408849345381613, 4.354298014093828, 1.79654673514508, -0.24252782534067613, -6.3266009356257795, -9.188741209956259, -15.679940427696112, -17.763638599037446, -22.47862856842786, -22.921734115807087, -11.090927883941962, -6.150513460631685, -2.5859551029299936, -0.1465864637679279, 1.174180008465187, 2.102254799749693, 0.34057563061352136, -0.9640689063008367, -4.552895226864808, -7.067788646968692, -10.864542381851585, -13.50913411206204, -14.848915651267513, -19.644327333786634, -21.918066822159002, -26.289601707151743, -27.851427160837467, -18.26329923323473, 0.715002120081876, 6.131642876815707, 7.410065076677268, 7.461327416026886, 7.023463505140629, 4.307683215932534, 2.776881220020983, -1.9200607216999046]
```
* tilt:
```
[82.12935920248833, 154.48529243115772, -125.9121980982381, -51.668860098809866, 31.30150561329588, 108.94322914871839, -163.85981305355207, -88.18863024060516, -27.38655794013396, 32.78989671359718, 94.60808759435002, 154.11340123465118, -142.60845733846688, -84.37262769023847, -18.483310322182067, 40.69239014134444, 105.41088837576473, 166.18691338599245, -125.51233845747394, -59.66764676092049, 12.542147860561524, 76.3493381469124, 125.3121009019235, 173.39652990478615, -137.37329118416446, -90.11541368629409, -39.87577834755655, 6.2275928955021, 58.20963762369964, 103.41059110323197, 157.6295285488888, -156.92883834961881, -108.29502508206761, -55.33608384369175, -4.884367818272769, 50.338653918224075, 104.89386454147606, 164.24857813665307, -142.92136782118382, -68.66826232106642, 5.6695651053340415, 82.73240550659735, 156.64433555913627, -124.45094691656672, -53.711736278845656, 27.23885574487593, 110.09525615495387]
```
Despite the negative value, they seem to be inaccurate or entirely wrong. 
So I thought of using a depth camera to convert the normalized coordinates to 3d points. Still, it is overcomplicated and usually is used with ROS, robotic operating system. I tried, but soon later, I stopped. 
Ultimately I decided to use Aruco marker to create a more precise environment placing them around the canvas to generate a 2D plane.

<img width="807" alt="aruco" src="https://user-images.githubusercontent.com/43594252/168498358-b9f0160e-51c0-4231-9127-3aef6423a9d3.png">

The results are:

* pan:
```
[0.49664780696470007, -0.27921784576290626, -0.17925150137411025, -0.483095154994407, -0.028372184502443464, -0.4696339896145002, 5.4043427089145704e-05, -0.666818065561771, -0.10010886689559616, 0.38453148259310466, -0.5285435134869176, 0.7139649376623908, -0.6694066641106721, 0.329032129000728, 0.49492378159840483, 0.7210918045217107, -0.37309862019383194, 0.6724732881959421, -0.16768387541015914, 0.39167247104987357, -0.12488996481658772, -0.535057883743008, -0.29230043016113866, 0.6419013463528099, -0.04712803759681042, -0.25898913000561763, -0.1440980784002982, -0.020168815780684222, 0.5825535653348957, 0.14995394047060634, 0.059338265562364996, -0.3345552392974085, 0.003211558670292691, -0.541993349536753, -0.4348667533272415, -0.4398057271927524, -0.2610601303124191, -0.008268305902178323, -0.7023457951710298, -0.6889448527161482, -0.7205308372677655, -0.5615486956802177, -0.6419528014890141, -0.24677933910166228, 0.030778623195293453, 0.28550287761561643, 0.5928395297467124]
```
* tilt:
```
[113.52228117304068, 63.71712573832663, 114.07259024668323, 94.46896262323037, 114.4084557358923, 56.18746074405451, 43.60979806126594, 140.85602342729473, 23.342649643000676, 59.779583389830606, 62.991221556698534, 39.02783220920663, 30.52011871971751, 101.33867937990672, 115.89795703592421, 111.13294486491232, 103.38607825766975, 124.15019306731463, 126.48540387723476, 38.27182745169824, 63.878257424528755, 51.82196939286727, 78.13195692344499, 158.76445361214437, 162.63177543385495, 128.70451591907369, 134.75530942844654, 2.499186049313427, 34.00640363430746, 131.26241016142717, 141.1521833859722, 135.2315466578959, 111.43987722332025, 111.91073845212152, 10.430046617782088, 5.663894574250857, 54.75582798154661, 52.73500971973644, 23.92008913371305, 64.52036411041536, 111.6198396829604, 113.47186656969266, 52.955384140266574, 57.64943215683514, 147.26390841981706, 143.1294897103298, 65.55353800137468]
```
Again despite the negative value seems to be more accurate, but not enught to draw with the robot.



