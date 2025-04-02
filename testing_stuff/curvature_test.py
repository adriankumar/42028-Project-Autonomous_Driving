#!/usr/bin/env python
import numpy as np
import h5py
import pygame
from skimage import transform as tf

# Initialize pygame
pygame.init()
size = (320*2, 160*2)
pygame.display.set_caption("Curvature Display")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
camera_surface = pygame.surface.Surface((320, 160), 0, 24).convert()

# ***** get perspective transform for images *****
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

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

# ***** functions to draw lines *****
def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return p2, p1

def draw_pt(img, x, y, color, sz=1):
    row, col = perspective_tform(x, y)
    # Convert to integers for slicing
    row, col = int(row), int(col)
    if row >= 0 and row < img.shape[0] and col >= 0 and col < img.shape[1]:
        img[row-sz:row+sz, col-sz:col+sz] = color

def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)

# ***** functions to calculate curvature *****
def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi/180.
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3   # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67    # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
    curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
    return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    # this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
    return y_actual, curvature

# ***** functions to draw paths *****
def draw_true_path(img, speed_ms, angle_steers):
    """Draw the true path based on log data (blue)"""
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, -angle_steers/10.0, path_x)
    draw_path(img, path_x, path_y, (0, 0, 255))  # Blue color

def draw_simulated_path(img, speed_ms, angle_steers):
    """Draw the simulated path based on external file (green)"""
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, -angle_steers, path_x)
    draw_path(img, path_x, path_y, (0, 255, 0))  # Green color

# ***** function to read steering angle from file *****
def read_steering_angle_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            angle = float(f.read().strip())
        return angle
    except (IOError, ValueError):
        return 0.0  # Default value if file can't be read

# ***** main loop *****
if __name__ == "__main__":
    # Load data
    dataset = "2016-06-08--11-46-01"
    log_file = f"./dataset/log/{dataset}.h5"
    cam_file = f"./dataset/camera/{dataset}.h5"
    
    log = h5py.File(log_file, "r")
    cam = h5py.File(cam_file, "r")
    
    angle_file = "./testing_stuff/angle_output/steering_angle.txt"
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    num_frames = log['times'].shape[0]
    i = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Get current frame - convert cam1_ptr to integer for indexing
        cam1_ptr_idx = int(log['cam1_ptr'][i])
        img = cam['X'][cam1_ptr_idx].swapaxes(0, 2).swapaxes(0, 1)
        
        # Get true steering angle and speed - convert to float
        true_angle = float(log['steering_angle'][i])
        speed_ms = float(log['speed'][i])
        
        # Get simulated steering angle from file
        sim_angle = read_steering_angle_from_file(angle_file)
        
        # Draw both paths
        draw_true_path(img, speed_ms, true_angle)      # Blue - true path
        draw_simulated_path(img, speed_ms, sim_angle)  # Green - simulated path
        
        # Display frame
        pygame.surfarray.blit_array(camera_surface, img.swapaxes(0, 1))
        camera_surface_2x = pygame.transform.scale2x(camera_surface)
        screen.blit(camera_surface_2x, (0, 0))
        pygame.display.flip()
        
        # Move to next frame (with looping)
        i = (i + 1) % num_frames
        
        # Cap framerate
        clock.tick(30)
    
    # Cleanup
    pygame.quit()
    cam.close()
    log.close()