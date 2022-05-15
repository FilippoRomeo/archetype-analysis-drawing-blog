# archetype-analysis-drawing-blog

*Archetype analysis* refers to the project of extrapolating the body's estimation poses from pictures and reproducing the output on a canvas. 
Archetype refers to Plato's ideas of pure mental forms imprinted in essence and encoded in a newborn individual. Later, Carl Jung used the term in
psychoanalysis, following the concept of undefined preforms that organise a structure that results in being intuitive in mental images. 
The main object of this project is to extrapolate, thanks to an AI, the archetype (pose) of dead bodies left from the war in Ukraine. Many websites allow
family members and friends to search, find and retrieve the dead body throughout those websites. Unfortunately, some are hardly identifiable as bodies due
to the brutality suffered. 

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

To extrapolate the pose estimation, I used the jetson nano, excellent hardware that contains 128 Cuda cores, allowing a DNN to run on real-time and images
and produce immediate results. 
Posenet returns two essential data a 2D array called *Links* and an object called *keypoints*. The 2D array stores information about the joins between body
parts ("links") needed to produce the skeletal topology. The keypoints object returns a list of items containing id and coordinates x and y required to
identify the body parts.


