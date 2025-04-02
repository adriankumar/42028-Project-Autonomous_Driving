# 42028-Project-Autonomous_Driving
 Project environment for developing autonomous driving AI system for 42028 Assignment 3

Currently in progress

GUI folder - contains the GUI official GUI scripts that will be modified during the project; contains the following:
- steering_wheel_gui: Simulating steering wheel movements through interface, the angles get written to a text file; TBI - where it reads angles from another text file to move the steering wheel (model predictions)
- steering_trajectory_calculator: external code file for calculating the steering trajectory display in the video gui annotations
- video_gui: application for comma.ai dataset visualisation (only works for comma.ai dataset)
- video_gui_flowchart: flow chart of how cideo_gui application works to track where to modify for scaled development

dataset folder - contains nothing, real dataset files are too large to store on github, they are there for keeping environment organised when pulled

testing_stuff folder - contains scripts to run to test environment set up such as:
- curvature_test: quick display test that steering curvature calculation works and can be displayed
- reading_h5_test: quick test to make sure dataset is downloaded properly and its contents can be accessed
- angle_output: this folder contains the text file that the steering angle from the steering wheel GUI is written to so that it can be read by the video GUI to display the simulated trajectory (green line; which will be model's prediction)