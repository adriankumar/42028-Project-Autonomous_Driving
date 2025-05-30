import numpy as np
from skimage import transform as tf

#calculations and display is taken exactly from comma ai's dataset information from the official github: https://github.com/commaai/research/blob/master/view_steering_model.py;
#however comments to explain in detail the calculations are still provided by us


#perspective transformation matrices
#these are calibrated points for the specific camera setup
#these won't work for other video data because the coordinate mapping of the road and the specific camera system is unknown
#so how these values were obtained, we don't know, they are fixed for the comma.ai dataset
rsrc = [
    [43.45456230828867, 118.00743250075844],
    [104.5055617352614, 69.46865203761757],
    [114.86050156739812, 60.83953551083698],
    [129.74572757609468, 50.48459567870026],
    [132.98164627363735, 46.38576532847949],
    [301.0336906326895, 98.16046448916306],
    [238.25686790036065, 62.56535881619311],
    [227.2547443287154, 56.30924933427718],
    [209.13359962247614, 46.817221154818526],
    [203.9561297064078, 43.5813024572758]
]
rdst = [
    [10.822125594094452, 1.42189132706374],
    [21.177065426231174, 1.5297552836484982],
    [25.275895776451954, 1.42189132706374],
    [36.062291434927694, 1.6376192402332563],
    [40.376849698318004, 1.42189132706374],
    [11.900765159942026, -2.1376192402332563],
    [22.25570499207874, -2.1376192402332563],
    [26.785991168638553, -2.029755283648498],
    [37.033067044190524, -2.029755283648498],
    [41.67121717733509, -2.029755283648498]
]

#initialise perspective transform matrix
tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

#convert road space coordinates to camera coordinates using the perspective transform
#transforms x y pixel value into rsrc coordinate system
def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return int(p2), int(p1)  #return as integers for array indexing

#by draw point, we mean, change the pixel value colour on the actual image to show the steering trajectory
#this is done for a single point x, note that x itself will be a list
def draw_pt(img, x, y, colour, sz=1):
    row, col = perspective_tform(x, y) #map to rsrc coordinates

    if row >= 0 and row < img.shape[0] and col >= 0 and col < img.shape[1]:
        img[row-sz:row+sz, col-sz:col+sz] = colour #colour the image

#draws the entire trajectory (all pixel values that have changed colours now create the trajectory)
#uses original coordinate values path_x, path_y which are lists
def draw_path(img, path_x, path_y, colour):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, colour)

#calculation of curvature points, some of these parameters are specific to the comma.ai dataset obtained from their data collection methodology
#that is unknown to us; uses only steering angle and speed
def calc_curvature(speed, steering_angle, angle_offset=0):
    deg_to_rad = np.pi/180.
    slip_factor = 0.0014  #slip factor from vehicle data
    steer_ratio = 15.3  #steering ratio for the vehicle
    wheel_base = 2.67  #vehicle wheelbase in meters
    
    steering_angle_rad = (steering_angle - angle_offset) * deg_to_rad #convert to radians for calculation
    curvature = steering_angle_rad/(steer_ratio * wheel_base * (1. + slip_factor * speed**2))
    return curvature

#y values in coordinate system, individual points that have a curvature based on car parameters to indicate pixel position based on distances list
def calc_path_offset(speed, steering_angle, distances, angle_offset=0):
    curvature = calc_curvature(speed, steering_angle, angle_offset)
    
    #clip curvature to avoid NaN issues with sharp turns
    offsets = distances * np.tan(np.arcsin(np.clip(distances * curvature, -0.999, 0.999))/2.)
    return offsets, curvature

#draw the true labels path
def draw_true_path(img, speed, steering_angle):
    distances = np.arange(0., 50.1, 0.5) #distances ahead in 0.5m increments
    offsets, _ = calc_path_offset(speed, -steering_angle, distances)
    draw_path(img, distances, offsets, (226, 32, 238)) #magenta

#draw the path simulated by the steering wheel gui, which still uses the same current speed that true labels does
def draw_simulated_path(img, speed, steering_angle):
    distances = np.arange(0., 50.1, 0.5) #distances ahead in 0.5m increments
    offsets, _ = calc_path_offset(speed, -steering_angle, distances)
    draw_path(img, distances, offsets, (69, 245, 177)) #cyan colour

#draw the annotated frame and return it, it's only called when the trajectory display is enabled
#draws both ground truth and simulated trajectories on the frame
def draw_trajectory_annotation(frame, speed, true_angle, sim_angle):
    #make a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    #draw true path (blue)
    draw_true_path(annotated_frame, speed, true_angle)
    
    #draw simulated path (green)
    draw_simulated_path(annotated_frame, speed, sim_angle)
    
    return annotated_frame