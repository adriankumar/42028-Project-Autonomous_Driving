import pygame
import math
import tkinter as tk
from PIL import Image, ImageTk

#colour constants
BG_COLOUR = "#2C2C30"  #background color for all panels
ACCEL_COLOR = "#a37bd1"  #purple for accelerator
ACCEL_PRESSED = "#7b5d9e"
BRAKE_COLOR = "#d67259"  #orange/coral for brake
BRAKE_PRESSED = "#944f3e"
WHEEL_COLOR = "#a37bd1"  #purple for steering wheel outline
SPOKES_COLOUR = "#4e3b6e" #lgith purple
HUB_COLOR = "#484850"  #slightly lighter than background
HUB_LOGO_COLOR = "#d67259"  #orange/coral for hub logo
TEXT_COLOR = "white"  #white text throughout

#creates accel or brake pedal widget for interactivity and simulation
class PedalControl:
    def __init__(self, parent_frame, width=100, height=60, pedal_type="accelerator", interactive=True):
        self.parent = parent_frame
        self.width = width
        self.height = height
        self.pedal_type = pedal_type  #"accelerator" or "brake"
        self.interactive = interactive
        
        #state tracking
        self.is_pressed = False
        self.press_depth = 0.0  #0 to 1 indicating how pressed the pedal is
        self.dragging = False
        self.last_mouse_y = 0
        self.is_running = True
        
        #colours based on pedal type
        if pedal_type == "accelerator":
            #convert hex to RGB tuple for pygame
            self.pedal_colour = tuple(int(ACCEL_COLOR[i:i+2], 16) for i in (1, 3, 5))
            self.pedal_pressed_colour = tuple(int(ACCEL_PRESSED[i:i+2], 16) for i in (1, 3, 5))  #darker purple when pressed
            self.label_text = "ACCEL"
        else:  #brake
            #convert hex to RGB tuple for pygame
            self.pedal_colour = tuple(int(BRAKE_COLOR[i:i+2], 16) for i in (1, 3, 5))
            self.pedal_pressed_colour = tuple(int(BRAKE_PRESSED[i:i+2], 16) for i in (1, 3, 5))  #darker coral when pressed
            self.label_text = "BRAKE"
        
        #convert background color to RGB tuple for pygame
        self.bg_colour = tuple(int(BG_COLOUR[i:i+2], 16) for i in (1, 3, 5))
        self.text_colour = (255, 255, 255)  #white
        
        #initialise pygame
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 12, bold=True)  #larger bold font
        
        #create pygame surface
        self.surface = pygame.Surface((self.width, self.height))
        
        #setup tkinter canvas - changed from pack to grid for better centering
        self.canvas = tk.Canvas(self.parent, width=self.width, height=self.height, 
                               highlightthickness=0, bg=BG_COLOUR)  #remove border with highlightthickness=0
        # self.canvas.pack(padx=5)  #Center the pedal in its container #dont pack the canvas, let the parent class pack it
        
        #bind mouse events if interactive
        if interactive:
            self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
            self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
            self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        
        #initial render
        self.render()
        self.update_canvas()
        
        #start animation loop
        self.update_animation()

    #------interactivity handlers------------
    def on_mouse_down(self, event):
        if not self.interactive:
            return
        self.dragging = True
        self.last_mouse_y = event.y
        self.is_pressed = True
    
    def on_mouse_up(self, event):
        self.dragging = False
        self.is_pressed = False
        self.press_depth = 0.0
    
    def on_mouse_move(self, event):
        if not self.interactive or not self.dragging:
            return
        
        #calculate press depth based on vertical movement
        delta_y = event.y - self.last_mouse_y
        self.press_depth = max(0.0, min(1.0, self.press_depth + delta_y / 30.0))
        self.last_mouse_y = event.y
    
    def set_pressed(self, pressed, depth=0.5):
        #external method to set pedal state
        self.is_pressed = pressed
        self.press_depth = depth if pressed else 0.0
    
    def set_interactive(self, interactive):
        self.interactive = interactive
    
    #------rendering/drawing------------
    def render(self):
        #fill background
        self.surface.fill(self.bg_colour)
        
        #calculate pressed offset
        pressed_offset = int(self.press_depth * 5) if self.is_pressed else 0
        
        #choose colour based on pressed state
        colour = self.pedal_pressed_colour if self.is_pressed else self.pedal_colour
        
        #draw pedal rectangle
        pedal_rect = pygame.Rect(10, 10 + pressed_offset, self.width - 20, self.height - 20)
        pygame.draw.rect(self.surface, colour, pedal_rect)
        pygame.draw.rect(self.surface, colour, pedal_rect, 2)  #outline same color
        
        #draw label in center of pedal
        label_surface = self.font.render(self.label_text, True, self.text_colour)
        label_rect = label_surface.get_rect(center=(self.width // 2, self.height // 2))
        self.surface.blit(label_surface, label_rect)
        
        #removed pressure indicator display
    
    def update_canvas(self):
        #convert pygame surface to tkinter format
        raw_data = pygame.image.tostring(self.surface, 'RGB')
        img = Image.frombytes('RGB', (self.width, self.height), raw_data)
        self.tk_img = ImageTk.PhotoImage(img)
        
        #update tkinter canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
    
    def update_animation(self):
        if self.is_running:
            self.render()
            self.update_canvas()
            self.parent.after(50, self.update_animation)
    
    def cleanup(self):
        self.is_running = False

#creates an interactive steering wheel interface that can be controlled by an angle value
class SteeringWheel:
    def __init__(self, parent_frame, width=200, height=200, interactive=True, label=""):
        self.parent = parent_frame
        self.width = width
        self.height = height
        self.label = label
        
        #state tracking
        self.current_angle = 0.0
        self.max_angle = 720 #two full rotations clockwise
        self.min_angle = -720 #two full rotations anti-clockwise
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.interactable = interactive
        self.is_running = True
        
        #colours - converted to RGB tuples for pygame
        self.bg_colour = tuple(int(BG_COLOUR[i:i+2], 16) for i in (1, 3, 5))
        self.wheel_colour = tuple(int(WHEEL_COLOR[i:i+2], 16) for i in (1, 3, 5))
        self.spokes_colour = tuple(int(SPOKES_COLOUR[i:i+2], 16) for i in (1, 3, 5))
        self.hub_colour = tuple(int(HUB_COLOR[i:i+2], 16) for i in (1, 3, 5))
        self.hub_logo_colour = tuple(int(HUB_LOGO_COLOR[i:i+2], 16) for i in (1, 3, 5))
        self.text_colour = (255, 255, 255)  #white text
        
        #wheel parameters
        self.wheel_center = (self.width // 2, self.height // 2)
        self.wheel_radius = min(self.width, self.height) // 3
        self.hub_radius = self.wheel_radius // 3
        self.logo_radius = self.hub_radius // 2
        self.spoke_width = 10
        
        #initialise pygame
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 12, bold=True)
        
        #create pygame surface
        self.surface = pygame.Surface((self.width, self.height))
        
        #setup tkinter canvas
        self.canvas = tk.Canvas(self.parent, width=self.width, height=self.height, 
                               highlightthickness=0, bg=BG_COLOUR)  #remove border with highlightthickness=0
        self.canvas.pack(expand=True, fill='both')
        
        #bind mouse events - now binding resize for ALL wheels, not just interactive ones
        self.canvas.bind("<Configure>", self.on_resize)
        
        if interactive:
            self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
            self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
            self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        
        #initial render
        self.render()
        self.update_canvas()
        
        #start animation loop
        self.update_animation()
    
    #------interactivity handlers------------
    def on_mouse_down(self, event):
        #check if interactable
        if not self.interactable:
            return
            
        mouse_pos = (event.x, event.y)
        
        if self.is_mouse_on_wheel(mouse_pos):
            self.dragging = True
            self.last_mouse_pos = mouse_pos
    
    def on_mouse_up(self, event):
        self.dragging = False
    
    def on_mouse_move(self, event):
        #check if interactable
        if not self.interactable or not self.dragging:
            return
            
        #update wheel angle
        mouse_pos = (event.x, event.y)
        self.update_wheel_angle(mouse_pos)
    
    def on_resize(self, event):
        #update dimensions
        self.width = event.width
        self.height = event.height
        
        #update wheel parameters based on new size
        self.wheel_center = (self.width // 2, self.height // 2)
        self.wheel_radius = min(self.width, self.height) // 3
        self.hub_radius = self.wheel_radius // 3
        self.logo_radius = self.hub_radius // 2
        self.spoke_width = max(2, min(10, self.wheel_radius // 10))  #adjust spoke width based on size
        
        #recreate surface
        self.surface = pygame.Surface((self.width, self.height))
        
        #re-render
        self.render()
        self.update_canvas()
    
    def is_mouse_on_wheel(self, mouse_pos):
        dx = mouse_pos[0] - self.wheel_center[0]
        dy = mouse_pos[1] - self.wheel_center[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance <= self.wheel_radius * 1.5
    
    #--------angle changes handling-----------
    def calculate_angle_change(self, new_pos):
        old_vector = (self.last_mouse_pos[0] - self.wheel_center[0], 
                     self.last_mouse_pos[1] - self.wheel_center[1])
        new_vector = (new_pos[0] - self.wheel_center[0], 
                     new_pos[1] - self.wheel_center[1])
        
        #calculate angles of bothvectors
        old_angle = math.atan2(old_vector[1], old_vector[0])
        new_angle = math.atan2(new_vector[1], new_vector[0])
        
        #calculate the difference (in degrees)
        delta_angle = math.degrees(new_angle - old_angle)
        
        #handle angle wrapping
        if delta_angle > 180:
            delta_angle -= 360
        elif delta_angle < -180:
            delta_angle += 360
            
        return delta_angle
    
    
    def update_wheel_angle(self, mouse_pos):
        if not self.dragging:
            return
        
        delta_angle = self.calculate_angle_change(mouse_pos)
        
        #update angle with constraints
        new_angle = self.current_angle + delta_angle
        new_angle = max(self.min_angle, min(new_angle, self.max_angle))
        
        self.current_angle = new_angle
        self.last_mouse_pos = mouse_pos
        
        #notify listeners of change
        self.on_angle_changed()
    
    #gets over written by child
    def on_angle_changed(self):
        pass
    
    def set_angle(self, angle):
        self.current_angle = max(self.min_angle, min(angle, self.max_angle))
        self.render()
        self.update_canvas()
    
    def set_interactable(self, interactable):
        self.interactable = interactable
    
    #--------rendering/drawing-----------
    def render(self):
        #fill background
        self.surface.fill(self.bg_colour)
        
        #draw outer wheel with new color
        pygame.draw.circle(self.surface, self.wheel_colour, self.wheel_center, self.wheel_radius, 4)
        
        #draw hub
        pygame.draw.circle(self.surface, self.hub_colour, self.wheel_center, self.hub_radius)
        pygame.draw.circle(self.surface, self.hub_logo_colour, self.wheel_center, self.logo_radius)
        
        #draws spokes
        angle_rad = math.radians(self.current_angle)
        for i in range(3):
            spoke_angle = angle_rad + i * (2 * math.pi / 3)
            spoke_start = (
                self.wheel_center[0] + self.hub_radius * math.cos(spoke_angle),
                self.wheel_center[1] + self.hub_radius * math.sin(spoke_angle)
            )
            spoke_end = (
                self.wheel_center[0] + self.wheel_radius * math.cos(spoke_angle),
                self.wheel_center[1] + self.wheel_radius * math.sin(spoke_angle)
            )
            pygame.draw.line(self.surface, self.spokes_colour, spoke_start, spoke_end, self.spoke_width)
        
        #draw angle display
        angle_text = f"{self.current_angle:.1f}°"
        text_surface = self.font.render(angle_text, True, self.text_colour)
        text_rect = text_surface.get_rect(center=(self.wheel_center[0], self.wheel_center[1] + self.wheel_radius + 20))
        self.surface.blit(text_surface, text_rect)
        
        #display label
        if self.label:
            label_surface = self.font.render(self.label, True, self.text_colour)
            label_rect = label_surface.get_rect(center=(self.wheel_center[0], self.wheel_center[1] - self.wheel_radius - 10))
            self.surface.blit(label_surface, label_rect)
        
        #indicate control state
        if self.label == "Simulated Driving":
            status_text = "User Control" if self.interactable else "Model Control"
            status_surface = self.font.render(status_text, True, self.text_colour)
            status_rect = status_surface.get_rect(center=(self.wheel_center[0], self.wheel_center[1] - self.wheel_radius - 25))
            self.surface.blit(status_surface, status_rect)
    
    def update_canvas(self):
        #convert pygame surface to tkinter format
        raw_data = pygame.image.tostring(self.surface, 'RGB')
        img = Image.frombytes('RGB', (self.width, self.height), raw_data)
        self.tk_img = ImageTk.PhotoImage(img)
        
        #update tkinter canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor=tk.NW)
    
    def update_animation(self):
        if self.is_running:
            self.render()
            self.update_canvas()
            self.parent.after(50, self.update_animation)
    
    def cleanup(self):
        #stop animation
        self.is_running = False

#steering control with pedals that is interactive and already pakced into a frame
class DrivingControl:
    def __init__(self, parent_frame, interactive=True, label=""):
        #main container frame with updated background color
        self.frame = tk.Frame(parent_frame, bg=BG_COLOUR)
        self.frame.pack(expand=True, fill='both')
        
        #create steering wheel frame
        self.steering_frame = tk.Frame(self.frame, bg=BG_COLOUR)
        self.steering_frame.pack(expand=True, fill='both')
        
        #create steering wheel
        self.steering_wheel = SteeringWheel(self.steering_frame, interactive=interactive, label=label)
        
        #create pedals frame with centering capability
        self.pedals_frame = tk.Frame(self.frame, bg=BG_COLOUR)
        self.pedals_frame.pack(fill='x', pady=5)
        
        #create a container to center the pedals horizontally
        self.pedals_container = tk.Frame(self.pedals_frame, bg=BG_COLOUR)
        self.pedals_container.pack(anchor=tk.CENTER)
        
        #create pedals side by side (horizontal)
        self.accelerator = PedalControl(self.pedals_container, pedal_type="accelerator", interactive=interactive)
        self.accelerator.canvas.pack(side=tk.LEFT, padx=5)
        
        self.brake = PedalControl(self.pedals_container, pedal_type="brake", interactive=interactive)
        self.brake.canvas.pack(side=tk.LEFT, padx=5)
        
        #state variables
        self.current_acceleration = 0.0
        self.callback = None
        
        #override callbacks
        self.steering_wheel.on_angle_changed = self.on_angle_changed
    
    def on_angle_changed(self):
        if self.callback:
            self.callback("steering", self.steering_wheel.current_angle)
    
    def set_steering_angle(self, angle):
        self.steering_wheel.set_angle(angle)
    
    def set_acceleration(self, accel_value):
        #determine which pedal to press based on acceleration value
        #increased dead zone threshold to reduce noise sensitivity
        dead_zone = 0.15
        
        #add hysteresis to prevent rapid pedal switching
        if not hasattr(self, 'last_pedal_state'):
            self.last_pedal_state = "none"  #initialize state tracking
            self.accel_smoothed = 0.0  #smoothed acceleration value
        
        #apply smoothing (exponential moving average)
        smoothing_factor = 0.3  #lower value = more smoothing
        self.accel_smoothed = (smoothing_factor * accel_value) + ((1 - smoothing_factor) * self.accel_smoothed)
        
        #determine pedal state with hysteresis
        if abs(self.accel_smoothed) < dead_zone:
            #no pedal pressed (inside dead zone)
            new_state = "none"
        elif self.accel_smoothed > 0:
            #acceleration
            new_state = "accelerator"
        else:
            #braking
            new_state = "brake"
        
        #add hysteresis to prevent rapid switching
        #only switch state if we've significantly crossed the threshold
        if self.last_pedal_state == "accelerator" and new_state == "brake" and self.accel_smoothed > -0.25:
            new_state = "none"  #maintain a gap between accel and brake
        elif self.last_pedal_state == "brake" and new_state == "accelerator" and self.accel_smoothed < 0.25:
            new_state = "none"  #maintain a gap between brake and accel
        
        #update pedals based on state
        if new_state == "none":
            self.accelerator.set_pressed(False)
            self.brake.set_pressed(False)
        elif new_state == "accelerator":
            #better normalization for observed data range
            press_depth = min(abs(self.accel_smoothed) / 0.7, 1.0)
            self.accelerator.set_pressed(True, press_depth)
            self.brake.set_pressed(False)
        else:  #brake
            #better normalization for observed data range
            press_depth = min(abs(self.accel_smoothed) / 0.7, 1.0)
            self.brake.set_pressed(True, press_depth)
            self.accelerator.set_pressed(False)
        
        #store state for next time
        self.last_pedal_state = new_state
        self.current_acceleration = accel_value  #keep original value for external reference
    
    def set_interactive(self, interactive):
        self.steering_wheel.set_interactable(interactive)
        self.accelerator.set_interactive(interactive)
        self.brake.set_interactive(interactive)
    
    def set_callback(self, callback):
        self.callback = callback
    
    def cleanup(self):
        self.steering_wheel.cleanup()
        self.accelerator.cleanup()
        self.brake.cleanup()

#creates true and simulated driving control objects packed into one frame, the main gui only needs to pass a parent frame that this simulator can be displayed and interacted with
class DualDrivingControlFrame:
    def __init__(self, parent_frame, callback=None):
        #main container frame with updated bg colour
        self.frame = tk.Frame(parent_frame, bg=BG_COLOUR)
        self.frame.pack(expand=True, fill='both')
        
        #create horizontal layout for driving controls
        self.controls_frame = tk.Frame(self.frame, bg=BG_COLOUR)
        self.controls_frame.pack(expand=True, fill='both')
        
        #create frames for each control side by side
        self.true_frame = tk.Frame(self.controls_frame, bg=BG_COLOUR)
        self.true_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        self.sim_frame = tk.Frame(self.controls_frame, bg=BG_COLOUR)
        self.sim_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        #configure grid weights
        self.controls_frame.columnconfigure(0, weight=1)
        self.controls_frame.columnconfigure(1, weight=1)
        self.controls_frame.rowconfigure(0, weight=1)
        
        #create driving controls
        self.true_control = DrivingControl(self.true_frame, interactive=False, label="True Driving")
        self.sim_control = DrivingControl(self.sim_frame, interactive=True, label="Simulated Driving")
        
        #callback
        self.callback = callback
        
        #set callback for simulated control
        self.sim_control.set_callback(self.on_sim_control_changed)
    
    #notify main gui of control change
    def on_sim_control_changed(self, control_type, value):
        if self.callback:
            self.callback(control_type, value)
    
    #update true values from telemetry data
    def set_true_values(self, steering_angle, acceleration):
        self.true_control.set_steering_angle(steering_angle)
        self.true_control.set_acceleration(acceleration)
    
    #update simulated values from user interaction or model predictions
    def set_sim_values(self, steering_angle, acceleration):
        self.sim_control.set_steering_angle(steering_angle)
        self.sim_control.set_acceleration(acceleration)
    
    #enable or disable the simulated control
    def set_interactable(self, interactable):
        self.sim_control.set_interactive(interactable)
    
    #returns current simulated values
    def get_sim_steering(self):
        return self.sim_control.steering_wheel.current_angle
    
    def get_sim_acceleration(self):
        return self.sim_control.current_acceleration
    
    #clean up both controls
    def cleanup(self):
        self.true_control.cleanup()
        self.sim_control.cleanup()