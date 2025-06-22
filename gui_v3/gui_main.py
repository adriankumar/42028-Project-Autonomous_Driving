import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import gui_alt.gui_backend as gb
from gui_alt.driving_sim import DualDrivingControlFrame
import gui_v3.model_handler as m
from gui_alt.speed_graph import SpeedGraph
from gui_v3.model_visualisation_gui import ModelVisualisation

#colour constants
# BG_COLOUR = "#1E1B2C"  #dark purple
BG_COLOUR = "#1A1A1C" #black with purple tint
# PANEL_COLOUR = "#2A263F" #lighter purple for general panels
PANEL_COLOUR = "#2C2C30" #greyish purple
DISPLAY_COLOUR = "black" #for video and saliency displays
TEXT_COLOUR = "white"
CONTROL_COLOUR = "#252140" #now using panel colour for driving controls

#button and slider colours
BUTTON_COLOUR = "#a37bd1"
BUTTON_ACTIVE_COLOUR = "#d67259"
SLIDER_TROUGH_COLOUR = "#433b4f"
SLIDER_BUTTON_COLOUR = "#d67259"
VIDEO_SLIDER_COLOUR = "#d67259"
AUG_SLIDER_COLOUR = "#d67259" 

#fixed dimensions
VIDEO_WIDTH = 864
VIDEO_HEIGHT = 432
SPACING = 10  #consistent spacing/padding

#main gui class for video display and model interaction
class VideoGui:
    def __init__(self, root):
        #store root
        self.root = root
        self.root.title("Driving Simulator Interface")
        self.root.configure(bg=BG_COLOUR)
        
        #state vars
        self.frames = None
        self.telemetry = None
        self.frame_count = 0
        self.current_idx = 0
        self.playing = False
        self.after_id = None
        self.fps = 20 #fps of video dataset is 20 fps
        self.updating_slider = False
        self.video_h = VIDEO_HEIGHT
        self.video_w = VIDEO_WIDTH
        
        #loop configuration variables
        self.loop_start_idx = 0
        self.loop_end_idx = 0
        self.loop_count = 0
        self.current_loop_iteration = 0
        self.loop_mode = False
        
        #saliency window - will be integrated into main layout instead of popup
        self.sal_window = None
        self.sal_lbl = None
        
        #steering angles and car acceleration
        self.true_steering_angle = 0.0
        self.sim_steering_angle = 0.0
        self.true_car_acceleration = 0.0
        self.sim_car_acceleration = 0.0

         #model speed simulation tracking
        self.model_sim_speed = 0.0  #speed used for model input (separate from display)
        self.model_context_frames = 0  #frames since model started
        self.context_build_threshold = 48  #frames of true speed before switching to simulated
        
        #model running
        self.model_running = False
        m.load_steering_model()

        #initialise the speed graph
        self.speed_graph_obj = SpeedGraph(fps=20) #frames were recorded at 20 fps according to comma ai
        
        #model visualisation
        self.model_visualisation = None
        self.model_vis_img = None
        
        #setup the main container with padding
        self.main_container = tk.Frame(self.root, bg=BG_COLOUR, padx=SPACING, pady=SPACING)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        #create the main layout grid - 3 rows, 3 columns
        self._setup_grid()
        self._create_widgets()
        
        #set a reasonable minimum size
        min_width = VIDEO_WIDTH + 400  #video width plus margin for side panels
        min_height = VIDEO_HEIGHT*2 + 120  #two displays plus controls and spacing
        self.root.minsize(min_width, min_height)

    #configure the 3x3 grid layout for main container
    def _setup_grid(self):
        #create the 3×3 grid layout (3 sections horizontally and vertically)
        #columns: left panel, center display, right panel
        self.main_container.columnconfigure(0, weight=1, minsize=200)  #left column - resizable with min width
        self.main_container.columnconfigure(1, weight=3)  #center column - now gets more space
        self.main_container.columnconfigure(2, weight=1, minsize=200)  #right column - resizable with min width
        
        #rows: top controls, main displays, bottom displays
        self.main_container.rowconfigure(0, weight=0)  #top controls - fixed height
        self.main_container.rowconfigure(1, weight=1)  #middle section - expands
        self.main_container.rowconfigure(2, weight=1)  #bottom section - expands
    
    #create all gui widgets and layout them in the grid
    def _create_widgets(self):
        #section 1: top row - fixed height controls
        self._create_top_controls()
        
        #section 2: middle row - video and left/right panels
        self._create_middle_row()
        
        #section 3: bottom row - saliency and panels
        self._create_bottom_row()

    #create a standard panel with app styling
    def _create_panel(self, parent, text, bg_colour=PANEL_COLOUR, height=None, width=None, bold=False):
        #create a standard panel with the app styling
        panel = tk.Frame(parent, bg=bg_colour, padx=SPACING, pady=SPACING)
        
        #set fixed dimensions if specified
        if height:
            panel.configure(height=height)
            panel.pack_propagate(False)
            
        if width:
            panel.configure(width=width)
            panel.pack_propagate(False)
            
        #create the label with optional bold
        font_style = ("Arial", 12, "bold") if bold else ("Arial", 12)
        label = tk.Label(panel, text=text, bg=bg_colour, fg=TEXT_COLOUR, font=font_style)
        label.pack(expand=True)
        
        return panel
    
    #create top control row with load button, playback controls and simulator title
    def _create_top_controls(self):
        #top row - all elements have the same height
        control_height = 70  #increased height to ensure slider fits
        
        #load file panel (left) - now just a centered button
        self.load_panel = tk.Frame(self.main_container, bg=PANEL_COLOUR, height=control_height)
        self.load_panel.grid(row=0, column=0, sticky="nsew", padx=(0,SPACING), pady=(0,SPACING))
        self.load_panel.pack_propagate(False)
        
        #add load button - centered and bold
        self.load_btn = tk.Button(self.load_panel, text="Load h5 file", bg=BUTTON_COLOUR, fg=TEXT_COLOUR,
                               activebackground=BUTTON_ACTIVE_COLOUR, command=self.load_file, 
                               font=("Arial", 11, "bold"), padx=15, pady=5)
        self.load_btn.pack(expand=True)
        
        #playback controls (center) - removed title, just buttons and slider
        self.playback_panel = tk.Frame(self.main_container, bg=PANEL_COLOUR, 
                                      height=control_height, width=VIDEO_WIDTH)
        self.playback_panel.grid(row=0, column=1, sticky="nsew", padx=SPACING, pady=(0,SPACING))
        self.playback_panel.pack_propagate(False)
        
        #create buttons
        self.button_frame = tk.Frame(self.playback_panel, bg=PANEL_COLOUR)
        self.button_frame.pack(side=tk.TOP, fill=tk.X, expand=True, pady=(5,0))
        
        #bold text for buttons
        self.play_btn = tk.Button(self.button_frame, text="Play", state="disabled", 
                              bg=BUTTON_COLOUR, fg=TEXT_COLOUR, activebackground=BUTTON_ACTIVE_COLOUR,
                              command=self.toggle_play, font=("Arial", 11, "bold"), padx=20, pady=3)
        self.play_btn.pack(side=tk.LEFT, expand=True, padx=5)
        
        self.model_btn = tk.Button(self.button_frame, text="Start Model", state="disabled",
                               bg=BUTTON_COLOUR, fg=TEXT_COLOUR, activebackground=BUTTON_ACTIVE_COLOUR,
                               command=self.toggle_model, font=("Arial", 11, "bold"), padx=20, pady=3)
        self.model_btn.pack(side=tk.LEFT, expand=True, padx=5)
        
        #slider in playback panel
        self.slider_frame = tk.Frame(self.playback_panel, bg=PANEL_COLOUR)
        self.slider_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=True, pady=(5,0))
        
        self.slider = tk.Scale(self.slider_frame, from_=0, to=0, orient=tk.HORIZONTAL, 
                          showvalue=False, state="disabled", command=self.slider_moved,
                          bg=PANEL_COLOUR, fg=TEXT_COLOUR, troughcolor=SLIDER_TROUGH_COLOUR,
                          activebackground=SLIDER_BUTTON_COLOUR, highlightthickness=0,
                          sliderrelief=tk.FLAT)
        self.slider.config(sliderlength=20)  #make slider button more visible
        self.slider.pack(fill=tk.X, expand=True, padx=5)
        
        #driving simulator title (right)
        self.simulator_title = self._create_panel(self.main_container, "Driving Simulator", height=control_height, bold=True)
        self.simulator_title.grid(row=0, column=2, sticky="nsew", padx=(SPACING,0), pady=(0,SPACING))
    
    #create middle row with left panels, video display and driving controls
    def _create_middle_row(self):
        #left panels (stacked in a frame)
        self.left_middle_frame = tk.Frame(self.main_container, bg=BG_COLOUR)
        self.left_middle_frame.grid(row=1, column=0, sticky="nsew", padx=(0,SPACING), pady=SPACING)
        
        #configure left frame for stacked panels
        self.left_middle_frame.columnconfigure(0, weight=1)
        self.left_middle_frame.rowconfigure(0, weight=1)
        self.left_middle_frame.rowconfigure(1, weight=1)
        
        #information panel (previously meta) - with bold title
        self.info_panel = self._create_panel(self.left_middle_frame, "Information", bold=True)
        self.info_panel.grid(row=0, column=0, sticky="nsew", pady=(0,SPACING/2))
        
        #add information content with larger font and bold categories
        self.info_frame = tk.Frame(self.info_panel, bg=PANEL_COLOUR)
        self.info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        #file label - category bold, value normal
        file_frame = tk.Frame(self.info_frame, bg=PANEL_COLOUR)
        file_frame.pack(fill=tk.X, anchor="w", pady=3)
        file_label = tk.Label(file_frame, text="File:", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11, "bold"), anchor="w")
        file_label.pack(side=tk.LEFT)
        self.file_lbl = tk.Label(file_frame, text="-", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11), anchor="w")
        self.file_lbl.pack(side=tk.LEFT, padx=(5,0))
        
        #frame label
        frame_frame = tk.Frame(self.info_frame, bg=PANEL_COLOUR)
        frame_frame.pack(fill=tk.X, anchor="w", pady=3)
        frame_label = tk.Label(frame_frame, text="Frame:", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11, "bold"), anchor="w")
        frame_label.pack(side=tk.LEFT)
        self.idx_lbl = tk.Label(frame_frame, text="0", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11), anchor="w")
        self.idx_lbl.pack(side=tk.LEFT, padx=(5,0))
        
        #time label
        time_frame = tk.Frame(self.info_frame, bg=PANEL_COLOUR)
        time_frame.pack(fill=tk.X, anchor="w", pady=3)
        time_label = tk.Label(time_frame, text="Time:", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11, "bold"), anchor="w")
        time_label.pack(side=tk.LEFT)
        self.time_lbl = tk.Label(time_frame, text="0.00s", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11), anchor="w")
        self.time_lbl.pack(side=tk.LEFT, padx=(5,0))
        
        #angle label
        angle_frame = tk.Frame(self.info_frame, bg=PANEL_COLOUR)
        angle_frame.pack(fill=tk.X, anchor="w", pady=3)
        angle_label = tk.Label(angle_frame, text="Angle:", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11, "bold"), anchor="w")
        angle_label.pack(side=tk.LEFT)
        self.angle_lbl = tk.Label(angle_frame, text="0.0°", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11), anchor="w")
        self.angle_lbl.pack(side=tk.LEFT, padx=(5,0))
        
        #accel label
        accel_frame = tk.Frame(self.info_frame, bg=PANEL_COLOUR)
        accel_frame.pack(fill=tk.X, anchor="w", pady=3)
        accel_label = tk.Label(accel_frame, text="Accel:", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11, "bold"), anchor="w")
        accel_label.pack(side=tk.LEFT)
        self.accel_lbl = tk.Label(accel_frame, text="0.0", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11), anchor="w")
        self.accel_lbl.pack(side=tk.LEFT, padx=(5,0))

        #context label
        context_frame = tk.Frame(self.info_frame, bg=PANEL_COLOUR)
        context_frame.pack(fill=tk.X, anchor="w", pady=3)
        context_label = tk.Label(context_frame, text="Context:", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11, "bold"), anchor="w")
        context_label.pack(side=tk.LEFT)
        self.context_lbl = tk.Label(context_frame, text="0/48", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                                font=("Arial", 11), anchor="w")
        self.context_lbl.pack(side=tk.LEFT, padx=(5,0))
        
        #options panel (bold title)
        self.options_panel = self._create_panel(self.left_middle_frame, "Options", bold=True)
        self.options_panel.grid(row=1, column=0, sticky="nsew", pady=(SPACING/2,0))
        
        #add options content
        self.options_frame = tk.Frame(self.options_panel, bg=PANEL_COLOUR)
        self.options_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.saliency_var = tk.IntVar(value=0)
        self.traj_var = tk.IntVar(value=0)
        self.fixed_ylim_var = tk.IntVar(value=1)  #default to fixed ylim
        self.show_accel_var = tk.IntVar(value=0)  #default to speed mode
        self.toggle_speed_sim_var = tk.IntVar(value=0)  #default unchecked
        self.model_vis_var = tk.IntVar(value=0)  #default unchecked
        
        self.saliency_cb = tk.Checkbutton(self.options_frame, text="Show Saliency Map", 
                                    variable=self.saliency_var, command=self.toggle_saliency,
                                    bg=PANEL_COLOUR, fg=TEXT_COLOUR, selectcolor=BUTTON_COLOUR,
                                    activebackground=PANEL_COLOUR, activeforeground=TEXT_COLOUR,
                                    font=("Arial", 11))
        self.saliency_cb.pack(anchor="w", fill=tk.X, pady=5)
        
        self.traj_cb = tk.Checkbutton(self.options_frame, text="Display Trajectory", 
                                variable=self.traj_var, command=self.update_display,
                                bg=PANEL_COLOUR, fg=TEXT_COLOUR, selectcolor=BUTTON_COLOUR,
                                activebackground=PANEL_COLOUR, activeforeground=TEXT_COLOUR,
                                font=("Arial", 11))
        self.traj_cb.pack(anchor="w", fill=tk.X, pady=5)
        
        #add toggle speed sim checkbox
        self.toggle_speed_sim_cb = tk.Checkbutton(self.options_frame, text="Toggle Speed Sim", 
                                        variable=self.toggle_speed_sim_var, command=self.toggle_speed_sim,
                                        bg=PANEL_COLOUR, fg=TEXT_COLOUR, selectcolor=BUTTON_COLOUR,
                                        activebackground=PANEL_COLOUR, activeforeground=TEXT_COLOUR,
                                        font=("Arial", 11))
        self.toggle_speed_sim_cb.pack(anchor="w", fill=tk.X, pady=5)
        
        #add model visualisation checkbox
        self.model_vis_cb = tk.Checkbutton(self.options_frame, text="Model Visualisation", 
                                        variable=self.model_vis_var, command=self.toggle_model_vis,
                                        bg=PANEL_COLOUR, fg=TEXT_COLOUR, selectcolor=BUTTON_COLOUR,
                                        activebackground=PANEL_COLOUR, activeforeground=TEXT_COLOUR,
                                        font=("Arial", 11))
        self.model_vis_cb.pack(anchor="w", fill=tk.X, pady=5)
        
        #add fixed ylim toggle - initially disabled
        self.fixed_ylim_cb = tk.Checkbutton(self.options_frame, text="Fixed Y-Axis Limits", 
                                        variable=self.fixed_ylim_var, command=self.toggle_fixed_ylim,
                                        bg=PANEL_COLOUR, fg=TEXT_COLOUR, selectcolor=BUTTON_COLOUR,
                                        activebackground=PANEL_COLOUR, activeforeground=TEXT_COLOUR,
                                        font=("Arial", 11), state="disabled")
        self.fixed_ylim_cb.pack(anchor="w", fill=tk.X, pady=5)
        
        #add acceleration mode toggle - initially disabled
        self.show_accel_cb = tk.Checkbutton(self.options_frame, text="Show Acceleration", 
                                        variable=self.show_accel_var, command=self.toggle_graph_mode,
                                        bg=PANEL_COLOUR, fg=TEXT_COLOUR, selectcolor=BUTTON_COLOUR,
                                        activebackground=PANEL_COLOUR, activeforeground=TEXT_COLOUR,
                                        font=("Arial", 11), state="disabled")
        self.show_accel_cb.pack(anchor="w", fill=tk.X, pady=5)
        
        #center video display frame - ensures perfect centering
        self.video_container = tk.Frame(self.main_container, bg=BG_COLOUR, width=VIDEO_WIDTH)
        self.video_container.grid(row=1, column=1, padx=SPACING, pady=SPACING)
        self.video_container.grid_propagate(False)  #maintain fixed width for container
        
        #create fixed-size video display centered in container - now with black placeholder
        self.video_display = tk.Frame(self.video_container, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, bg=DISPLAY_COLOUR)
        self.video_display.pack(expand=True)
        self.video_display.pack_propagate(False)  #maintain fixed size
        
        #add video label
        self.video_lbl = tk.Label(self.video_display)
        self.video_lbl.pack(expand=True, fill=tk.BOTH)
        
        #initialize with black
        black = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        self.show_image(black, self.video_lbl)
        
        #create driving controls panel on right
        self.driving_controls_panel = tk.Frame(self.main_container, bg=PANEL_COLOUR, padx=SPACING, pady=SPACING)
        self.driving_controls_panel.grid(row=1, column=2, sticky="nsew", padx=(SPACING,0), pady=SPACING)
        
        #add driving controls
        self.driving_controls = DualDrivingControlFrame(self.driving_controls_panel, callback=self.on_sim_driving_changed)
    

    #toggle between fixed and dynamic y-axis limits for speed graph
    def toggle_fixed_ylim(self):
        #toggle between fixed and dynamic y-axis limits
        if hasattr(self, 'speed_graph_obj'):
            self.speed_graph_obj.fixed_ylim = (self.fixed_ylim_var.get() == 1)
            self.update_speed_graph()
    
    #toggle between speed and acceleration display modes for graph
    def toggle_graph_mode(self):
        #toggle between speed and acceleration display modes
        if hasattr(self, 'speed_graph_obj'):
            #set the display mode based on checkbox
            self.speed_graph_obj.mode = "acceleration" if self.show_accel_var.get() == 1 else "speed"
            
            #reset data when switching modes
            self.speed_graph_obj.frames = []
            self.speed_graph_obj.true_speeds = []
            self.speed_graph_obj.sim_speeds = []
            self.speed_graph_obj.last_frame = -1
            
            #reset min/max observed when switching modes
            self.speed_graph_obj.min_observed = float('inf')
            self.speed_graph_obj.max_observed = float('-inf')
            self.speed_graph_obj.zoom_enabled = False
            
            #update to refresh graph display
            self.update_speed_graph()

    #create bottom row with augmentation controls, saliency display and visualisation panel
    def _create_bottom_row(self):
        #left augment panel with bold title
        self.augment_panel = self._create_panel(self.main_container, "Augmentation", bold=True)
        self.augment_panel.grid(row=2, column=0, sticky="nsew", padx=(0,SPACING), pady=SPACING)
        
        #add augmentation content
        self.aug_frame = tk.Frame(self.augment_panel, bg=PANEL_COLOUR)
        self.aug_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        #light slider - bold label
        light_label = tk.Label(self.aug_frame, text="Light", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11, "bold"))
        light_label.pack(anchor="w", pady=(5,0))
        
        self.light_slider = tk.Scale(self.aug_frame, from_=0.0, to=0.5, resolution=0.1, orient=tk.HORIZONTAL,
                                command=self.on_augmentation_changed, bg=PANEL_COLOUR, fg=TEXT_COLOUR,
                                troughcolor=SLIDER_TROUGH_COLOUR, activebackground=AUG_SLIDER_COLOUR,
                                highlightthickness=0, font=("Arial", 9), sliderrelief=tk.FLAT)
        self.light_slider.pack(fill=tk.X)
        self.light_slider.config(sliderlength=20)  #make slider button more visible
        
        #dim slider - bold label
        dim_label = tk.Label(self.aug_frame, text="Dim", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                        font=("Arial", 11, "bold"))
        dim_label.pack(anchor="w", pady=(5,0))
        
        self.dim_slider = tk.Scale(self.aug_frame, from_=0.0, to=0.5, resolution=0.1, orient=tk.HORIZONTAL,
                            command=self.on_augmentation_changed, bg=PANEL_COLOUR, fg=TEXT_COLOUR,
                            troughcolor=SLIDER_TROUGH_COLOUR, activebackground=AUG_SLIDER_COLOUR,
                            highlightthickness=0, font=("Arial", 9), sliderrelief=tk.FLAT)
        self.dim_slider.pack(fill=tk.X)
        self.dim_slider.config(sliderlength=20)  #make slider button more visible
        
        #noise slider - bold label
        noise_label = tk.Label(self.aug_frame, text="Noise", bg=PANEL_COLOUR, fg=TEXT_COLOUR, 
                            font=("Arial", 11, "bold"))
        noise_label.pack(anchor="w", pady=(5,0))
        
        self.noise_slider = tk.Scale(self.aug_frame, from_=0.0, to=0.5, resolution=0.1, orient=tk.HORIZONTAL,
                                command=self.on_augmentation_changed, bg=PANEL_COLOUR, fg=TEXT_COLOUR,
                                troughcolor=SLIDER_TROUGH_COLOUR, activebackground=AUG_SLIDER_COLOUR,
                                highlightthickness=0, font=("Arial", 9), sliderrelief=tk.FLAT)
        self.noise_slider.pack(fill=tk.X)
        self.noise_slider.config(sliderlength=20)  #make slider button more visible
        
        #center saliency display frame - ensures perfect centering
        self.saliency_container = tk.Frame(self.main_container, bg=BG_COLOUR, width=VIDEO_WIDTH)
        self.saliency_container.grid(row=2, column=1, padx=SPACING, pady=SPACING)
        self.saliency_container.grid_propagate(False)  #maintain fixed width for container
        
        #create fixed-size saliency display centered in container - now embedded in main layout
        self.saliency_display = tk.Frame(self.saliency_container, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, bg=DISPLAY_COLOUR)
        self.saliency_display.pack(expand=True)
        self.saliency_display.pack_propagate(False)  #maintain fixed size
        
        #add saliency label
        self.saliency_lbl = tk.Label(self.saliency_display)
        self.saliency_lbl.pack(expand=True, fill=tk.BOTH)
        
        #initialise with black
        black = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        self.show_image(black, self.saliency_lbl)
        
        #right visualisation panel - contains multiple frames that can be switched
        self.vis_panel = self._create_panel(self.main_container, "Visualisation", bold=True)
        self.vis_panel.grid(row=2, column=2, sticky="nsew", padx=(SPACING,0), pady=SPACING)
        
        #create container for switching between different visualisations
        self.vis_container = tk.Frame(self.vis_panel, bg=PANEL_COLOUR)
        self.vis_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        #default placeholder frame
        self.default_vis_frame = tk.Frame(self.vis_container, bg=PANEL_COLOUR)
        default_label = tk.Label(self.default_vis_frame, text="Toggle a visualisation", 
                            bg=PANEL_COLOUR, fg=TEXT_COLOUR, font=("Arial", 14, "bold"))
        default_label.pack(expand=True)
        self.default_vis_frame.pack(fill=tk.BOTH, expand=True)
        
        #speed graph frame - initially hidden
        self.speed_graph_frame = tk.Frame(self.vis_container, bg=PANEL_COLOUR, width=500, height=300)
        self.speed_graph_lbl = tk.Label(self.speed_graph_frame, bg=PANEL_COLOUR)
        self.speed_graph_lbl.pack(fill=tk.BOTH, expand=True)
        
        #initialise with blank graph
        blank_img = self.speed_graph_obj.get_image()
        self.speed_graph_img = ImageTk.PhotoImage(blank_img)
        self.speed_graph_lbl.config(image=self.speed_graph_img)
        
        #model visualisation frame - initially hidden
        self.model_vis_frame = tk.Frame(self.vis_container, bg=PANEL_COLOUR)
        self.model_vis_lbl = tk.Label(self.model_vis_frame, bg=PANEL_COLOUR)
        self.model_vis_lbl.pack(fill=tk.BOTH, expand=True)
        
        #track current visualisation
        self.current_vis = "default"

    #update speed graph display with current data
    def update_speed_graph(self):
        #only update if speed sim is enabled
        if self.toggle_speed_sim_var.get() == 0 or self.telemetry is None:
            return
        
        #determine which mode to display based on checkbox
        if self.show_accel_var.get() == 0:  #speed mode
            #get current true speed
            current_true_speed = gb.get_speed(self.telemetry, self.current_idx)
            
            #update speed graph
            if self.model_running:
                #use the actual simulated speed that the model is using for decisions
                #after context building, otherwise use true speed
                if self.model_context_frames >= self.context_build_threshold:
                    display_sim_speed = self.model_sim_speed
                else:
                    display_sim_speed = current_true_speed
                
                #update with true speed and actual model simulated speed
                speed_img = self.speed_graph_obj.update(
                    self.current_idx, 
                    current_true_speed,
                    display_sim_speed,
                    self.model_running
                )
            else:
                #only show true speed when model is not running
                speed_img = self.speed_graph_obj.update(
                    self.current_idx, 
                    current_true_speed,
                    model_running=False
                )
        else:  #acceleration mode - unchanged
            current_true_accel = self.true_car_acceleration
            
            if self.model_running:
                speed_img = self.speed_graph_obj.update(
                    self.current_idx, 
                    current_true_accel,
                    self.sim_car_acceleration,
                    self.model_running
                )
            else:
                speed_img = self.speed_graph_obj.update(
                    self.current_idx, 
                    current_true_accel,
                    model_running=False
                )
        
        #update display with fixed size image
        self.speed_graph_img = ImageTk.PhotoImage(speed_img)
        self.speed_graph_lbl.config(image=self.speed_graph_img)

    #update model visualisation display with current neuron states
    def update_model_visualisation(self):
        #update model visualisation if enabled and available
        if self.model_visualisation and self.current_vis == "model":
            try:
                #get hidden states from model handler
                if hasattr(m, 'hidden_state') and m.hidden_state is not None:
                    #get synaptic weights from model
                    synaptic_weights = m.get_synaptic_weights()
                    
                    #update visualisation with current hidden states and synaptic weights
                    self.model_visualisation.update_neuron_states(m.hidden_state, synaptic_weights)
                    
                    #generate updated image from existing figure
                    vis_img = self.model_visualisation.get_visualisation_image()
                    
                    #convert to tkinter format and display
                    vis_photo = ImageTk.PhotoImage(vis_img)
                    self.model_vis_lbl.config(image=vis_photo)
                    self.model_vis_img = vis_photo  #keep reference to prevent garbage collection
                    
            except Exception as e:
                print(f"error updating model visualisation: {e}")

    #handle changes in simulated driving controls from user interaction
    def on_sim_driving_changed(self, control_type, value):
        #update simulated values based on control type
        if control_type == "steering":
            self.sim_steering_angle = value
            #update angle display
            self.angle_lbl.config(text=f'{value:.1f}°')
        elif control_type == "acceleration":
            self.sim_car_acceleration = value
        
        #update trajectory if enabled
        if self.traj_var.get() == 1 and self.frames is not None:
            #redisplay current frame to update trajectory
            self.update_display()
    
    #display numpy array image in specified label widget
    def show_image(self, img_arr, label_widget):
        img = Image.fromarray(img_arr)
        img = img.resize((self.video_w, self.video_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label_widget.photo = photo  #keeps reference to prevent garbage collection
        label_widget.config(image=photo)

    #load h5 video file and corresponding telemetry data
    def load_file(self):
        path = filedialog.askopenfilename(title='Select h5 file', filetypes=[('h5 files', '*.h5')])
        if not path:
            return
        if not gb.is_h5(path):
            messagebox.showerror('Error', 'Invalid file type')
            return
        try:
            self.frames, self.telemetry = gb.load_camera_and_labels(path)
        except Exception as e:
            messagebox.showerror('Error', str(e))
            return
        
        #after loading a new file, reset the speed grap
        if hasattr(self, 'speed_graph_obj'):
            blank_img = self.speed_graph_obj.reset()
            self.speed_graph_img = ImageTk.PhotoImage(blank_img)
            self.speed_graph_lbl.config(image=self.speed_graph_img)
        
        self.frame_count = gb.get_frame_count(self.frames)
        self.current_idx = 0
        self.play_btn.config(state='normal')
        self.slider.config(state='normal', to=self.frame_count-1)
        self.model_btn.config(state='normal')
        self.file_lbl.config(text=f'{path.split("/")[-1]}')
        self.display_frame(0)

    #handle augmentation slider changes by updating display
    def on_augmentation_changed(self, *args):
        #update display when augmentation sliders change
        if self.frames is not None:
            self.update_display()

    #refresh display with current frame and settings
    def update_display(self):
        #update display with current frame and settings
        if self.frames is not None:
            self.display_frame(self.current_idx)

    #display specific frame with all current settings applied
    def display_frame(self, idx):
        #get augmentation values
        light_val = float(self.light_slider.get())
        dim_val = float(self.dim_slider.get())
        noise_val = float(self.noise_slider.get())
        
        #get processed frame with augmentations and trajectory if enabled
        frame = gb.process_frame(
            self.frames, 
            idx, 
            self.telemetry, 
            self.true_steering_angle, 
            self.sim_steering_angle,
            self.traj_var.get() == 1,
            light_val,
            dim_val,
            noise_val
        )
        
        #display the processed frame
        self.show_image(frame, self.video_lbl)
        self.idx_lbl.config(text=f'{idx}')
        
        #update time information
        time_val = self.telemetry['times'][idx] if self.telemetry and 'times' in self.telemetry else idx/self.fps
        self.time_lbl.config(text=f'{time_val:.2f}s')
        
        #update true steering angle and acceleration
        if self.telemetry and 'steering_angle' in self.telemetry:
            angle = self.telemetry['steering_angle'][idx]
            self.true_steering_angle = angle
            
            #update acceleration
            accel = gb.get_car_acceleration(self.telemetry, idx)
            self.true_car_acceleration = accel
            
            #set true values in driving controls
            self.driving_controls.set_true_values(angle, accel)
        
        #update slider with error handling to prevent getting stuck
        try:
            if not self.updating_slider:
                self.updating_slider = True
                self.root.after_idle(lambda: self.slider.set(idx))
                self.updating_slider = False
        except Exception:
            self.updating_slider = False
            
        #update saliency if enabled
        if self.saliency_var.get() == 1:
            self.update_saliency_display()
        
        #update context display if model is running
        if self.model_running:
            context_text = f"{min(self.model_context_frames, self.context_build_threshold)}/{self.context_build_threshold}"
            if hasattr(self, 'context_lbl'):
                self.context_lbl.config(text=context_text)
        
        self.update_speed_graph()
        self.update_model_visualisation()

    #handle manual slider movement by user
    def slider_moved(self, val):
        if self.updating_slider or self.frames is None:
            return
        self.current_idx = int(float(val))
        self.display_frame(self.current_idx)

    #toggle play/pause for regular video playback
    def toggle_play(self):
        if not self.playing:
            self.playing = True
            self.play_btn.config(text='Pause')
            self.play_next()
        else:
            self.playing = False
            self.play_btn.config(text='Play')
            if self.after_id:
                self.root.after_cancel(self.after_id)

    #advance to next frame during regular playback
    def play_next(self):
        if not self.playing:
            return
        if self.current_idx >= self.frame_count-1:
            self.toggle_play()
            return
        self.current_idx += 1
        self.display_frame(self.current_idx)
        
        delay = int(1000/self.fps)
        self.after_id = self.root.after(delay, self.play_next)

    #show popup dialog to configure loop parameters
    def show_loop_config_dialog(self):
        #create modal dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Loop Configuration")
        dialog.configure(bg=BG_COLOUR)
        dialog.geometry("300x250")
        dialog.resizable(False, False)
        
        #center dialog on parent window
        x = self.root.winfo_rootx() + 50
        y = self.root.winfo_rooty() + 50
        dialog.geometry(f"+{x}+{y}")
        
        #make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()
        
        #result variable to track dialog outcome
        result = {'success': False}
        
        #create input fields
        tk.Label(dialog, text="Start Index:", bg=BG_COLOUR, fg=TEXT_COLOUR, font=("Arial", 11)).pack(pady=5)
        start_entry = tk.Entry(dialog, font=("Arial", 11))
        start_entry.pack(pady=5)
        start_entry.insert(0, str(self.current_idx))
        
        tk.Label(dialog, text="End Index:", bg=BG_COLOUR, fg=TEXT_COLOUR, font=("Arial", 11)).pack(pady=5)
        end_entry = tk.Entry(dialog, font=("Arial", 11))
        end_entry.pack(pady=5)
        end_entry.insert(0, str(min(self.current_idx + 100, self.frame_count - 1)))
        
        tk.Label(dialog, text="Number of Loops:", bg=BG_COLOUR, fg=TEXT_COLOUR, font=("Arial", 11)).pack(pady=5)
        loops_entry = tk.Entry(dialog, font=("Arial", 11))
        loops_entry.pack(pady=5)
        loops_entry.insert(0, "5")
        
        #button frame
        button_frame = tk.Frame(dialog, bg=BG_COLOUR)
        button_frame.pack(pady=20)
        
        #ok button handler
        def ok_clicked():
            try:
                start_idx = int(start_entry.get())
                end_idx = int(end_entry.get())
                loop_count = int(loops_entry.get())
                
                #validate inputs
                if (start_idx < end_idx and 
                    0 <= start_idx < self.frame_count and 
                    0 <= end_idx < self.frame_count and
                    loop_count > 0):
                    
                    #store configuration
                    self.loop_start_idx = start_idx
                    self.loop_end_idx = end_idx
                    self.loop_count = loop_count
                    self.current_loop_iteration = 0
                    result['success'] = True
                    dialog.destroy()
                else:
                    messagebox.showerror("Invalid Input", "Please check your inputs")
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid numeric values")
        
        #cancel button handler
        def cancel_clicked():
            dialog.destroy()
        
        #create buttons with proper tk prefix
        tk.Button(button_frame, text="OK", command=ok_clicked, bg=BUTTON_COLOUR, fg=TEXT_COLOUR, 
                  font=("Arial", 11), padx=20).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel_clicked, bg=PANEL_COLOUR, fg=TEXT_COLOUR,
                  font=("Arial", 11), padx=20).pack(side=tk.LEFT, padx=5)
        
        #wait for dialog to close
        self.root.wait_window(dialog)
        
        return result['success']

    #toggle model running state and handle loop configuration
    def toggle_model(self):
        #toggle model runningstate
        if not self.model_running:
            #show loop configuration dialog before starting
            if not self.show_loop_config_dialog():
                return  #user cancelled, don't start model
            
            #start the model with loop configuration
            self.model_running = True
            self.loop_mode = True
            self.model_btn.config(text='Stop Model')
            self.driving_controls.set_interactable(False)
            
            #disable play button and slider during model run
            self.play_btn.config(state='disabled')
            self.slider.config(state='disabled')
            
            m.reset_model_state() #reset hidden state
            
            #reset model simulation tracking
            self.model_context_frames = 0
            self.model_sim_speed = 0.0
            self.current_loop_iteration = 0
            
            #set starting position
            self.current_idx = self.loop_start_idx
            
            #start playback similar to play button functionality
            if self.after_id:
                self.root.after_cancel(self.after_id)

            self.model_play_next()

        else:
            #stop the model
            self.model_running = False
            self.loop_mode = False
            self.model_btn.config(text='Start Model')
            self.driving_controls.set_interactable(True)
            
            #re-enable play button and slider
            self.play_btn.config(state='normal')
            self.slider.config(state='normal')
            
            #stop playback
            if self.after_id:
                self.root.after_cancel(self.after_id)

            m.reset_model_state() #reset hidden state

    #advance to next frame during model playback with loop handling
    def model_play_next(self):
        if not self.model_running:
            return
            
        #check if we've reached the end of the loop
        if self.current_idx >= self.loop_end_idx:
            #increment loop iteration counter
            self.current_loop_iteration += 1
            
            #check if we've completed all loops
            if self.current_loop_iteration >= self.loop_count:
                self.toggle_model()  #stop model
                return
            
            #reset to start of loop for next iteration
            self.current_idx = self.loop_start_idx
            #reset model state for clean loop iteration
            m.reset_model_state()
            self.model_context_frames = 0
            self.model_sim_speed = 0.0
        else:
            self.current_idx += 1
            
        #track frame context building
        
        #get augmented frame without trajectory for model input
        light_val = float(self.light_slider.get())
        dim_val = float(self.dim_slider.get())
        noise_val = float(self.noise_slider.get())
        
        #get augmented frame without trajectory for model inference
        augmented_frame = gb.process_frame(
            self.frames, 
            self.current_idx, 
            self.telemetry, 
            self.true_steering_angle, 
            self.sim_steering_angle,
            False,  #show_trajectory=False
            light_val,
            dim_val,
            noise_val
        )
        
        #determine which speed to use for model input
        if self.model_context_frames < self.context_build_threshold:
            #use true speed during context building phase
            model_input_speed = gb.get_speed(self.telemetry, self.current_idx)
            #also initialise simulated speed to current true speed
            self.model_sim_speed = model_input_speed
            self.model_context_frames += 1 #increment count of accumulated context frames
        else:
            #use simulated speed after context is built
            model_input_speed = self.model_sim_speed
        
        #predict steering angle and acceleration using the appropriate speed
        predicted_steering, predicted_accel = m.predict_steering_and_acceleration(
            augmented_frame, model_input_speed, normalise=True
        )
        
        #update simulated values if predictions available
        if predicted_steering is not None and predicted_accel is not None:
            self.sim_steering_angle = predicted_steering
            self.sim_car_acceleration = predicted_accel
            self.driving_controls.set_sim_values(predicted_steering, predicted_accel)
            self.angle_lbl.config(text=f'{predicted_steering:.1f}°')
            self.accel_lbl.config(text=f'{predicted_accel:.2f}')
            
            #update simulated speed based on acceleration prediction (only after context building)
            if self.model_context_frames >= self.context_build_threshold:
                time_step = 1.0 / self.fps
                self.model_sim_speed = max(0.0, self.model_sim_speed + (predicted_accel * time_step))
        
        #display the frame (this will apply trajectory visualization if enabled)
        self.display_frame(self.current_idx)
        
        delay = int(1000/self.fps)
        self.after_id = self.root.after(delay, self.model_play_next)

    #handle speed simulation toggle for graph display
    def toggle_speed_sim(self):
        #handle speed simulation toggle
        if self.toggle_speed_sim_var.get() == 1:
            #speed sim enabled - uncheck model vis if checked
            if self.model_vis_var.get() == 1:
                self.model_vis_var.set(0)
            
            #enable speed graph controls
            self.fixed_ylim_cb.config(state="normal")
            self.show_accel_cb.config(state="normal")
            
            #switch to speed graph visualisation
            self.switch_visualisation("speed")
        else:
            #speed sim disabled - disable controls and switch to default
            self.fixed_ylim_cb.config(state="disabled")
            self.show_accel_cb.config(state="disabled")
            self.switch_visualisation("default")

    #handle model visualisation toggle
    def toggle_model_vis(self):
        #handle model visualisation toggle
        if self.model_vis_var.get() == 1:
            #model vis enabled - uncheck speed sim if checked
            if self.toggle_speed_sim_var.get() == 1:
                self.toggle_speed_sim_var.set(0)
                #disable speed graph controls
                self.fixed_ylim_cb.config(state="disabled")
                self.show_accel_cb.config(state="disabled")
            
            #initialise model visualisation if not already done
            if self.model_visualisation is None:
                try:
                    #get model from model handler
                    if m.model is not None:
                        self.model_visualisation = ModelVisualisation(m.model)
                        print("model visualisation initialised successfully")
                    else:
                        print("model not loaded in model handler")
                        self.model_vis_var.set(0)  #uncheck if model not available
                        return
                except Exception as e:
                    print(f"error initialising model visualisation: {e}")
                    self.model_vis_var.set(0)  #uncheck if initialisation fails
                    return
            
            #enable synaptic weight capture
            m.enable_synaptic_weight_capture()
            
            #switch to model visualisation
            self.switch_visualisation("model")
            
            #update visualisation immediately
            self.update_model_visualisation()
        else:
            #model vis disabled - disable synaptic weight capture and switch to default
            m.disable_synaptic_weight_capture()
            self.switch_visualisation("default")

    #switch between different visualisation frames in right panel
    def switch_visualisation(self, vis_type):
        #switch between different visualisation frames
        if self.current_vis == vis_type:
            return
        
        #hide current frame
        if self.current_vis == "speed":
            self.speed_graph_frame.pack_forget()
        elif self.current_vis == "model":
            self.model_vis_frame.pack_forget()
        elif self.current_vis == "default":
            self.default_vis_frame.pack_forget()
        
        #show new frame
        if vis_type == "speed":
            self.speed_graph_frame.pack(fill=tk.BOTH, expand=True)
        elif vis_type == "model":
            self.model_vis_frame.pack(fill=tk.BOTH, expand=True)
        else:  #default
            self.default_vis_frame.pack(fill=tk.BOTH, expand=True)
        
        self.current_vis = vis_type

    #toggle saliency map display on/off
    def toggle_saliency(self):
        if self.saliency_var.get() == 1:
            #update saliency display if model is running
            self.update_saliency_display()
        else:
            #clear saliency display
            black = np.zeros((self.video_h, self.video_w, 3), dtype=np.uint8)
            self.show_image(black, self.saliency_lbl)

    #update saliency map display with current frame
    def update_saliency_display(self):
        #only update if enabled and frames exist
        if self.saliency_var.get() == 1 and self.frames is not None:
            light_val = float(self.light_slider.get())
            dim_val = float(self.dim_slider.get())
            noise_val = float(self.noise_slider.get())
            
            augmented_frame = gb.process_frame(
                self.frames, 
                self.current_idx, 
                self.telemetry, 
                self.true_steering_angle, 
                self.sim_steering_angle,
                False,  #show_trajectory=False
                light_val,
                dim_val,
                noise_val
            )
            
            #only generate saliency map if model is running
            if self.model_running:
                saliency_map = m.generate_saliency_map(augmented_frame, normalise=True)
                if saliency_map is not None:
                    self.show_image(saliency_map, self.saliency_lbl)
            else:
                #display black image if model not running
                black = np.zeros_like(augmented_frame)
                self.show_image(black, self.saliency_lbl)

    #cleanup resources when closing application
    def cleanup(self):
        #cleanup pygame resources
        if hasattr(self, 'driving_controls'):
            self.driving_controls.cleanup()
        
        #stop playback
        if self.playing:
            self.toggle_play()
        
        #cancel any pending after callbacks
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        
        #cleanup matplotlib resources
        if hasattr(self, 'speed_graph_obj'):
            self.speed_graph_obj.cleanup()
        
        #cleanup model visualisation
        if hasattr(self, 'model_visualisation') and self.model_visualisation:
            self.model_visualisation.cleanup()
        
        #clear references to images to help garbage collection
        if hasattr(self, 'speed_graph_img'):
            self.speed_graph_img = None
        
        if hasattr(self, 'model_vis_img'):
            self.model_vis_img = None
        
        if hasattr(self, 'video_lbl') and self.video_lbl:
            self.video_lbl.photo = None
        
        if hasattr(self, 'saliency_lbl') and self.saliency_lbl:
            self.saliency_lbl.photo = None