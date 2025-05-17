import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
import numpy as np
import gui.gui_backend as gb
from gui.driving_sim import DualDrivingControlFrame
import gui.model_handler as m

#builds the entire gui system and handles data passing, state variables and interaction handlers
class VideoGui:
    def __init__(self, root):
        #store root
        self.root = root
        self.root.title('driving gui')

        #state vars - entire gui uses these state vars to track real time processing and prevent multiple event handlers to conflict with each other
        self.frames = None
        self.telemetry = None
        self.frame_count = 0
        self.current_idx = 0
        self.playing = False
        self.after_id = None
        self.fps = 20 #fps of video dataset is 20 fps
        self.updating_slider = False
        self.video_h = 360
        self.video_w = 640
        
        #saliency window
        self.sal_window = None
        self.sal_lbl = None
        
        #steering angles and car acceleration
        self.true_steering_angle = 0.0
        self.sim_steering_angle = 0.0
        self.true_car_acceleration = 0.0
        self.sim_car_acceleration = 0.0
        
        #model running
        self.model_running = False
        m.load_steering_model()

        #build layout
        self.create_layout()
        # self.display_black()

#------------building widgets and frames helper functions------------
    def create_layout(self):
        #configure grid
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)
        
        #load files frame
        self.load_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.load_frame.grid(row=0, column=0, sticky='nw')
        self.build_load_widgets()
        
        #playback frame
        self.play_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.play_frame.grid(row=0, column=1, sticky='nsew')
        self.build_play_widgets()
        
        #steering frame container - now has driving controls
        self.steer_frame = tk.Frame(self.root, bg='#e0e0e0', width=200)
        self.steer_frame.grid(row=0, column=2, rowspan=2, sticky='nsew')
        self.build_steering_widgets()
        
        #meta frame
        self.meta_frame = tk.Frame(self.root, bg='#f8f8f8')
        self.meta_frame.grid(row=1, column=0, sticky='nsew')
        self.build_meta_widgets()
        
        #video container
        self.video_container = tk.Frame(self.root, bg='#000000')
        self.video_container.grid(row=1, column=1, sticky='nsew')
        self.video_container.columnconfigure(0, weight=1)
        self.video_container.rowconfigure(0, weight=1) 
        self.build_video_widgets()

    def build_load_widgets(self):
        self.load_btn = tk.Button(self.load_frame, text='load file', command=self.load_file)
        self.load_btn.pack(padx=6, pady=6)

    def build_play_widgets(self):
        #button row
        btn_row = tk.Frame(self.play_frame, bg='#f0f0f0')
        btn_row.pack(fill='x')
        #create centered container
        center_frame = tk.Frame(btn_row, bg='#f0f0f0')
        center_frame.pack(expand=True, anchor='center')
        self.play_btn = tk.Button(center_frame, text='play', state='disabled', command=self.toggle_play)
        self.play_btn.pack(side='left', padx=4, pady=4)
        self.model_btn = tk.Button(center_frame, text='start model', state='disabled', command=self.toggle_model) 
        self.model_btn.pack(side='left', padx=4, pady=4)
        #slider row
        slider_row = tk.Frame(self.play_frame, bg='#f0f0f0')
        slider_row.pack(fill='x')
        slider_center = tk.Frame(slider_row, bg='#f0f0f0')
        slider_center.pack(expand=True, anchor='center')
        self.slider = tk.Scale(slider_center, from_=0, to=0, orient='horizontal', command=self.slider_moved, state='disabled', showvalue=False, length=400)
        self.slider.pack(padx=8)

    def build_meta_widgets(self):
        info_frame = tk.LabelFrame(self.meta_frame, text='information')
        info_frame.pack(fill='x', padx=6, pady=4)
        self.file_lbl = tk.Label(info_frame, text='file: -')
        self.file_lbl.pack(anchor='w')
        self.idx_lbl = tk.Label(info_frame, text='frame: 0')
        self.idx_lbl.pack(anchor='w')
        self.time_lbl = tk.Label(info_frame, text='time: 0.00s')
        self.time_lbl.pack(anchor='w')
        self.angle_lbl = tk.Label(info_frame, text='angle: 0.0°')
        self.angle_lbl.pack(anchor='w')
        self.accel_lbl = tk.Label(info_frame, text='accel: 0.0')
        self.accel_lbl.pack(anchor='w')
        
        opt_frame = tk.LabelFrame(self.meta_frame, text='options')
        opt_frame.pack(fill='x', padx=6, pady=4)
        self.saliency_var = tk.IntVar(value=0)
        self.traj_var = tk.IntVar(value=0)
        sal_cb = tk.Checkbutton(opt_frame, text='saliency map', variable=self.saliency_var, command=self.toggle_saliency)
        sal_cb.pack(anchor='w')
        traj_cb = tk.Checkbutton(opt_frame, text='display trajectory', variable=self.traj_var, command=self.update_display)
        traj_cb.pack(anchor='w')
        
        aug_frame = tk.LabelFrame(self.meta_frame, text='augmentation')
        aug_frame.pack(fill='x', padx=6, pady=4)

        #light slider
        light_label = tk.Label(aug_frame, text='light')
        light_label.pack(anchor='w')
        self.light_slider = tk.Scale(aug_frame, from_=0.0, to=0.5, resolution=0.1, orient='horizontal', command=self.on_augmentation_changed)
        self.light_slider.pack(fill='x')

        #dim slider
        dim_label = tk.Label(aug_frame, text='dim')
        dim_label.pack(anchor='w')
        self.dim_slider = tk.Scale(aug_frame, from_=0.0, to=0.5, resolution=0.1, orient='horizontal', command=self.on_augmentation_changed)
        self.dim_slider.pack(fill='x')

        #noise slider
        noise_label = tk.Label(aug_frame, text='noise')
        noise_label.pack(anchor='w')
        self.noise_slider = tk.Scale(aug_frame, from_=0.0, to=0.5, resolution=0.1, orient='horizontal', command=self.on_augmentation_changed)
        self.noise_slider.pack(fill='x')

    def build_video_widgets(self):
        self.video_frame = tk.Frame(self.video_container, width=self.video_w, height=self.video_h, bg='#000000')
        self.video_frame.grid(row=0, column=0, sticky='nsew')
        self.video_frame.grid_propagate(False)
        self.video_lbl = tk.Label(self.video_frame)
        self.video_lbl.pack(expand=True, fill='both')

    def build_steering_widgets(self):
        #create driving control frame
        self.driving_controls = DualDrivingControlFrame(self.steer_frame, callback=self.on_sim_driving_changed)

#---------------------------------------------------------------------
    def on_sim_driving_changed(self, control_type, value):
        #update simulated values based on control type
        if control_type == "steering":
            self.sim_steering_angle = value
            #update angle display
            self.angle_lbl.config(text=f'angle: {value:.1f}°')
        elif control_type == "acceleration":
            self.sim_car_acceleration = value
        
        #update trajectory if enabled
        if self.traj_var.get() == 1 and self.frames is not None:
            #redisplay current frame to update trajectory
            self.update_display()

#--------------functionalitiies-------------
    def show_image(self, img_arr, label_widget):
        img = Image.fromarray(img_arr)
        img = img.resize((self.video_w, self.video_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label_widget.photo = photo  #keeps reference to prevent garbage collection
        label_widget.config(image=photo)

    def load_file(self):
        path = filedialog.askopenfilename(title='select h5 file', filetypes=[('h5 files', '*.h5')])
        if not path:
            return
        if not gb.is_h5(path):
            messagebox.showerror('error', 'invalid file type')
            return
        try:
            self.frames, self.telemetry = gb.load_camera_and_labels(path)
        except Exception as e:
            messagebox.showerror('error', str(e))
            return
        
        self.frame_count = gb.get_frame_count(self.frames)
        self.current_idx = 0
        self.play_btn.config(state='normal')
        self.slider.config(state='normal', to=self.frame_count-1)
        self.model_btn.config(state='normal')
        self.file_lbl.config(text=f'file: {path.split("/")[-1]}')
        self.display_frame(0)
#---------------------------------------------------------------------
#--------------Display and interaction handlers-------------------------
    def on_augmentation_changed(self, *args):
        #update display when augmentation sliders change
        if self.frames is not None:
            self.update_display()

    def update_display(self):
        #update display with current frame and settings
        if self.frames is not None:
            self.display_frame(self.current_idx)

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
        self.idx_lbl.config(text=f'frame: {idx}')
        
        #update time information
        time_val = self.telemetry['times'][idx] if self.telemetry and 'times' in self.telemetry else idx/self.fps
        self.time_lbl.config(text=f'time: {time_val:.2f}s')
        
        #update true steering angle and acceleration
        if self.telemetry and 'steering_angle' in self.telemetry:
            angle = self.telemetry['steering_angle'][idx]
            self.true_steering_angle = angle
            
            #update acceleration
            accel = gb.get_car_acceleration(self.telemetry, idx)
            self.true_car_acceleration = accel
            
            #set true values in driving controls
            self.driving_controls.set_true_values(angle, accel)
        
        #update slider
        if not self.updating_slider:
            self.updating_slider = True
            self.slider.set(idx)
            self.updating_slider = False
            
        #update saliency if required
        if self.saliency_var.get()==1 and self.sal_window and self.sal_window.winfo_exists():
            #only update saliency here if model is not running
            #when model is running, saliency is updated in model_play_next
            if not self.model_running:
                black = np.zeros((self.video_h, self.video_w, 3), dtype=np.uint8)
                self.show_image(black, self.sal_lbl)

    def slider_moved(self, val):
        if self.updating_slider or self.frames is None:
            return
        self.current_idx = int(float(val))
        self.display_frame(self.current_idx)

    def toggle_play(self):
        if not self.playing:
            self.playing = True
            self.play_btn.config(text='pause')
            self.play_next()
        else:
            self.playing = False
            self.play_btn.config(text='play')
            if self.after_id:
                self.root.after_cancel(self.after_id)

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

    def toggle_model(self):
        #toggle model runningstate
        self.model_running = not self.model_running
        
        if self.model_running:
            #start the model
            self.model_btn.config(text='stop model')
            self.driving_controls.set_interactable(False)
            
            #disable play button and slider during model run
            self.play_btn.config(state='disabled')
            self.slider.config(state='disabled')
            
            m.reset_model_state() #reset hidden state
            
            #start playback similar to play button functionality
            if self.after_id:
                self.root.after_cancel(self.after_id)
            self.model_play_next()
        else:
            #stop the model
            self.model_btn.config(text='start model')
            self.driving_controls.set_interactable(True)
            
            #re-enable play button and slider
            self.play_btn.config(state='normal')
            self.slider.config(state='normal')
            
            #stop playback
            if self.after_id:
                self.root.after_cancel(self.after_id)
            m.reset_model_state() #reset hidden state

    def model_play_next(self):
        if not self.model_running:
            return
        if self.current_idx >= self.frame_count-1:
            self.toggle_model()  #stop when reaching the end
            return
            
        self.current_idx += 1
        
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
        
        #get current speed for model input
        current_speed = gb.get_speed(self.telemetry, self.current_idx)
        
        #predict steering angle and acceleration from augmented frame
        predicted_steering, predicted_accel = m.predict_steering_and_acceleration(
            augmented_frame, current_speed, normalise=True
        )
        
        #update simulated values if predictions available
        if predicted_steering is not None and predicted_accel is not None:
            self.sim_steering_angle = predicted_steering
            self.sim_car_acceleration = predicted_accel
            self.driving_controls.set_sim_values(predicted_steering, predicted_accel)
            self.angle_lbl.config(text=f'angle: {predicted_steering:.1f}°')
            self.accel_lbl.config(text=f'accel: {predicted_accel:.2f}')
        
        #display the frame (this will apply trajectory visualization if enabled)
        self.display_frame(self.current_idx)
        
        #update saliency map if enabled
        self.update_saliency_display()
        
        delay = int(1000/self.fps)
        self.after_id = self.root.after(delay, self.model_play_next)

#----------------saliency map displays--------------------
    def create_saliency_window(self):
        if self.sal_window is None or not self.sal_window.winfo_exists():
            self.sal_window = Toplevel(self.root)
            self.sal_window.title('saliency map')
            self.sal_window.geometry(f'{self.video_w}x{self.video_h}')
            self.sal_window.resizable(False, False)

            #add label
            self.sal_lbl = tk.Label(self.sal_window)
            self.sal_lbl.pack(expand=True, fill='both')

            #set close behavior
            self.sal_window.protocol("WM_DELETE_WINDOW", self.handle_saliency_close)

            #create black image placeholder
            black = np.zeros((self.video_h, self.video_w, 3), dtype=np.uint8)
            self.show_image(black, self.sal_lbl)

    def handle_saliency_close(self):
        #update checkbox state
        self.saliency_var.set(0)

        #hide window
        if self.sal_window and self.sal_window.winfo_exists():
            self.sal_window.withdraw()

    def toggle_saliency(self):
        if self.saliency_var.get()==1:
            #show saliency in popup window
            self.create_saliency_window()

            #show window
            self.sal_window.deiconify()
            
            #update saliency display if model is running
            if self.model_running and self.frames is not None:
                self.update_saliency_display()
        else:
            #hide saliency window
            if self.sal_window and self.sal_window.winfo_exists():
                self.sal_window.withdraw()

    def update_saliency_display(self):
        #only update if saliency window exists and is visible
        if (self.sal_window and self.sal_window.winfo_exists() and 
            self.saliency_var.get()==1 and self.frames is not None):
            
            #get augmented frame for saliency processing (same as for model input)
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
                    self.show_image(saliency_map, self.sal_lbl)
            else:
                #display black image if model not running
                black = np.zeros_like(augmented_frame)
                self.show_image(black, self.sal_lbl)

#---------------------------------------------------------------------
    def cleanup(self):
        #cleanup pygame resources
        if hasattr(self, 'driving_controls'):
            self.driving_controls.cleanup()
        
        #stop playback
        if self.playing:
            self.toggle_play()