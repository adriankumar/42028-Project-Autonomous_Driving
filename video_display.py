import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import h5py
import numpy as np
import os
from PIL import Image, ImageTk
import time
import threading

#GUI FOR DISPLAYING DATASET VIDEOS (only works for h5 files from comma.ai dataset)
#select the h5 file from the camera directory (which will automatically load the corresponding log file with the same name in log directory)
#the GUI will display the video alongside all its telemetry data playing live

class H5VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("H5 Video Player")
        
        #initialise variables
        self.camera_file = None
        self.log_file = None
        self.frames = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.is_playing = False
        self.play_thread = None
        self.fps = 20  #default fps to 20Hz as per dataset description
        self.updating_slider = False  #flag to prevent recursive calls
        self.metric_visibility = {}  #individual metric visibility state
        
        #metrics to display with descriptions, taken from the log file keys
        self.metrics_info = {
            'blinker': {'description': 'Turn signal status', 'format': 'bool'},
            'brake': {'description': 'Combined brake signal', 'format': 'bool'},
            'brake_computer': {'description': 'Commanded brake [0-4095]', 'format': 'int'},
            'brake_user': {'description': 'User brake pedal depression [0-4095]', 'format': 'int'},
            'car_accel': {'description': 'm/s^2, from derivative of wheel speed', 'format': 'float2'},
            'fiber_accel': {'description': 'm/s^2', 'format': 'vector'},
            'fiber_gyro': {'description': 'deg/s', 'format': 'vector'},
            'gas': {'description': 'Throttle position [0-1]', 'format': 'float2'},
            'gear_choice': {'description': '0=park/neutral, 10=reverse, 11=changing', 'format': 'gear'},
            'selfdrive': {'description': 'Autonomous mode active', 'format': 'bool'},
            'speed': {'description': 'm/s, negative in reverse', 'format': 'float2'},
            'speed_abs': {'description': 'm/s absolute value', 'format': 'float2'},
            'speed_fl': {'description': 'Front-left wheel speed (m/s)', 'format': 'float2'},
            'speed_fr': {'description': 'Front-right wheel speed (m/s)', 'format': 'float2'},
            'speed_rl': {'description': 'Rear-left wheel speed (m/s)', 'format': 'float2'},
            'speed_rr': {'description': 'Rear-right wheel speed (m/s)', 'format': 'float2'},
            'standstill': {'description': 'Is the car stopped?', 'format': 'bool'},
            'steering_angle': {'description': 'Steering wheel angle (deg)', 'format': 'angle'},
            'steering_torque': {'description': 'Steering angle rate (deg/s)', 'format': 'float2'},
            'times': {'description': 'Timestamp (seconds)', 'format': 'float3'}
        }
        
        #initialise visibility for all metrics (default hidden)
        for metric in self.metrics_info:
            self.metric_visibility[metric] = tk.BooleanVar(value=False)
        
        #list of metrics to display (prioritised order)
        self.metrics_list = list(self.metrics_info.keys())
        self.metrics_data = {}  #dictionary to store loaded metrics
        self.cam1_ptr = None  #array to store frame pointer indices
        
        #set initial window size
        self.root.geometry("1100x700")
        
        #create ui components
        self.create_ui()
        
        #bind window resize event to update video display
        self.root.bind("<Configure>", self.on_window_resize)
    
    def create_ui(self):
        #create main frames
        self.control_frame = tk.Frame(self.root, padx=10, pady=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.video_frame = tk.Frame(self.main_frame, bg="black", width=640, height=320)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.video_frame.pack_propagate(False)  #prevent frame from shrinking
        
        self.info_frame = tk.LabelFrame(self.main_frame, text="Information", font=("Arial", 12, "bold"), padx=10, pady=10, width=380)
        self.info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        #create file selection button
        self.load_btn = tk.Button(self.control_frame, text="Load H5 File", command=self.load_h5_file, font=("Arial", 10, "bold"))
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        #create playback controls
        self.play_btn = tk.Button(self.control_frame, text="Play", command=self.play_video, font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = tk.Button(self.control_frame, text="Pause", command=self.pause_video, font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = tk.Button(self.control_frame, text="Reset", command=self.reset_video, font=("Arial", 10, "bold"), state=tk.DISABLED)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        #create fps control
        fps_frame = tk.Frame(self.control_frame)
        fps_frame.pack(side=tk.RIGHT, padx=5)
        tk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="20")  #set default to 20
        fps_spinbox = tk.Spinbox(fps_frame, from_=1, to=60, width=3, textvariable=self.fps_var, 
                               command=self.update_fps)
        fps_spinbox.pack(side=tk.LEFT)
        
        #create frame slider
        self.slider_frame = tk.Frame(self.root, padx=10, pady=5)
        self.slider_frame.pack(fill=tk.X)
        
        self.frame_slider = ttk.Scale(self.slider_frame, from_=0, to=1, orient=tk.HORIZONTAL, command=self.on_slider_change)
        self.frame_slider.pack(fill=tk.X, padx=10)
        self.frame_slider.state(['disabled'])
        
        #create video display
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        #info panel with three sections: basic info, toggle buttons, and metrics display
        
        #1. Basic info section
        self.info_header_frame = ttk.Frame(self.info_frame)
        self.info_header_frame.pack(fill=tk.X, pady=5)
        
        self.filename_label = tk.Label(self.info_header_frame, text="File: None", font=("Arial", 10), anchor="w", wraplength=340)
        self.filename_label.pack(fill=tk.X)
        
        self.filesize_label = tk.Label(self.info_header_frame, text="Size: 0 GB", font=("Arial", 10), anchor="w")
        self.filesize_label.pack(fill=tk.X)
        
        self.frame_label = tk.Label(self.info_header_frame, text="Frame: 0/0", font=("Arial", 10), anchor="w")
        self.frame_label.pack(fill=tk.X)
        
        ttk.Separator(self.info_frame, orient='horizontal').pack(fill='x', pady=5)
        
        #2. Toggle buttons section
        self.toggle_frame = ttk.LabelFrame(self.info_frame, text="Toggle Metrics")
        self.toggle_frame.pack(fill=tk.X, pady=5, padx=2)
        
        #Create toggle buttons frame with scrollbar for many metrics
        self.toggle_canvas = tk.Canvas(self.toggle_frame, height=100)
        self.toggle_scrollbar = ttk.Scrollbar(self.toggle_frame, orient="vertical", command=self.toggle_canvas.yview)
        self.toggle_buttons_frame = ttk.Frame(self.toggle_canvas)
        
        self.toggle_buttons_frame.bind(
            "<Configure>",
            lambda e: self.toggle_canvas.configure(scrollregion=self.toggle_canvas.bbox("all"))
        )
        
        self.toggle_canvas.create_window((0, 0), window=self.toggle_buttons_frame, anchor="nw")
        self.toggle_canvas.configure(yscrollcommand=self.toggle_scrollbar.set)
        
        #Create toggle buttons (organised in a grid)
        self.toggle_buttons = {}
        self.create_toggle_buttons()
        
        #Pack toggle canvas and scrollbar
        self.toggle_canvas.pack(side="left", fill="both", expand=True)
        self.toggle_scrollbar.pack(side="right", fill="y")
        
        ttk.Separator(self.info_frame, orient='horizontal').pack(fill='x', pady=5)
        
        #3. Metrics display with scrollbar
        self.metrics_frame = ttk.LabelFrame(self.info_frame, text="Metrics")
        self.metrics_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=2)
        
        self.metrics_canvas = tk.Canvas(self.metrics_frame)
        self.metrics_scrollbar = ttk.Scrollbar(self.metrics_frame, orient="vertical", command=self.metrics_canvas.yview)
        self.metrics_container = ttk.Frame(self.metrics_canvas)
        
        self.metrics_container.bind(
            "<Configure>",
            lambda e: self.metrics_canvas.configure(scrollregion=self.metrics_canvas.bbox("all"))
        )
        
        self.metrics_canvas.create_window((0, 0), window=self.metrics_container, anchor="nw")
        self.metrics_canvas.configure(yscrollcommand=self.metrics_scrollbar.set)
        
        #Create metrics displays (initially empty since all toggles are off)
        self.metric_frames = {}
        
        #Pack metrics canvas and scrollbar
        self.metrics_canvas.pack(side="left", fill="both", expand=True)
        self.metrics_scrollbar.pack(side="right", fill="y")
    
    def create_toggle_buttons(self):
        #Create toggle buttons in a multi-column grid layout
        columns = 2  #number of columns for toggle buttons
        row, col = 0, 0
        
        for metric in self.metrics_list:
            #Create styled toggle button
            toggle_button = ttk.Checkbutton(
                self.toggle_buttons_frame,
                text=metric.replace('_', ' ').title(),
                variable=self.metric_visibility[metric],
                command=lambda m=metric: self.toggle_metric(m),
                padding=2
            )
            toggle_button.grid(row=row, column=col, sticky="w", padx=2, pady=2)
            self.toggle_buttons[metric] = toggle_button
            
            #Move to next position
            col += 1
            if col >= columns:
                col = 0
                row += 1
    
    def create_metrics_display(self):
        #Clear existing metrics
        for widget in self.metrics_container.winfo_children():
            widget.destroy()
            
        self.metric_frames = {}
        
        #Create metric frames for metrics that should be visible
        for metric in self.metrics_list:
            if self.metric_visibility[metric].get():
                self.metric_frames[metric] = self.create_metric_widget(
                    self.metrics_container,
                    metric,
                    self.metrics_info[metric]['description'],
                    "N/A"
                )
    
    def create_metric_widget(self, parent, name, description, initial_value):
        #Frame to hold the metric
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=4, padx=2)
        
        #Add a subtle separator line
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=1)
        
        #Metric name in bold
        metric_name = name.replace('_', ' ').title()
        name_label = tk.Label(frame, text=f"{metric_name}", 
                           font=("Arial", 10, "bold"), anchor="w")
        name_label.pack(fill=tk.X)
        
        #Value and description in a sub-frame
        value_frame = ttk.Frame(frame)
        value_frame.pack(fill=tk.X, pady=2)
        
        value_label = tk.Label(value_frame, text=initial_value, 
                           font=("Arial", 10), anchor="w")
        value_label.pack(side=tk.LEFT, padx=(10, 5))
        
        desc_label = tk.Label(value_frame, text=f"({description})", 
                          font=("Arial", 8), fg="gray", wraplength=320, 
                          justify=tk.LEFT)
        desc_label.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        return {
            "frame": frame,
            "value": value_label,
            "name": name_label,
            "description": desc_label,
        }
    
    def toggle_metric(self, metric_name):
        is_visible = self.metric_visibility[metric_name].get()
        
        if is_visible and metric_name not in self.metric_frames:
            #Add the metric to the display
            self.metric_frames[metric_name] = self.create_metric_widget(
                self.metrics_container,
                metric_name,
                self.metrics_info[metric_name]['description'],
                "N/A"
            )
            
            #Update the value if we have data
            if self.frames is not None:
                aligned_metrics = self.get_aligned_metrics(self.current_frame_idx)
                if metric_name in aligned_metrics:
                    value = aligned_metrics[metric_name]
                    display_value = self.format_raw_value(value)
                    self.metric_frames[metric_name]["value"].config(text=display_value)
        
        elif not is_visible and metric_name in self.metric_frames:
            #Remove the metric from display
            self.metric_frames[metric_name]["frame"].destroy()
            del self.metric_frames[metric_name]
    
    def toggle_all_metrics(self, show=True):
        for metric, var in self.metric_visibility.items():
            if var.get() != show:
                var.set(show)
                self.toggle_metric(metric)
    
    def on_window_resize(self, event):
        #only update if we have a video loaded and it's the main window being resized
        if event.widget == self.root and self.frames is not None:
            #redisplay current frame to fit new size
            self.display_frame(self.current_frame_idx)
    
    def update_fps(self):
        try:
            self.fps = int(self.fps_var.get())
        except ValueError:
            self.fps = 20  #fallback to default 20fps
            self.fps_var.set("20")
    
    def load_h5_file(self):
        #stop any ongoing playback
        self.pause_video()
        
        #open file dialog
        file_path = filedialog.askopenfilename(
            title="Select H5 File",
            filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return  #user cancelled
        
        #check if file is h5
        if not file_path.lower().endswith('.h5'):
            messagebox.showerror("Error", "File is not h5")
            return
        
        try:
            #close any open files
            if self.camera_file is not None:
                self.camera_file.close()
                self.camera_file = None
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            
            #reset metrics data
            self.metrics_data = {}
            
            #get filename and directory
            filename = os.path.basename(file_path)
            directory = os.path.dirname(file_path)
            
            #check if it's a camera or log file
            if "camera" in directory.lower():
                #it's a camera file, find corresponding log file
                camera_path = file_path
                log_path = file_path.replace("camera", "log")
            elif "log" in directory.lower():
                #it's a log file, find corresponding camera file
                log_path = file_path
                camera_path = file_path.replace("log", "camera")
            else:
                #can't determine, try to open as a camera file
                camera_path = file_path
                log_path = None
            
            #try to open camera file
            if camera_path and os.path.exists(camera_path):
                try:
                    self.camera_file = h5py.File(camera_path, 'r')
                    if 'X' not in self.camera_file:
                        messagebox.showerror("Error", "Camera file doesn't contain expected data")
                        self.camera_file.close()
                        self.camera_file = None
                        return
                    self.frames = self.camera_file['X']
                    self.total_frames = self.frames.shape[0]
                except Exception as e:
                    messagebox.showerror("Error", f"Error opening camera file: {str(e)}")
                    return
            else:
                messagebox.showerror("Error", "Camera file not found")
                return
            
            #try to open log file and load metrics
            if log_path and os.path.exists(log_path):
                try:
                    self.log_file = h5py.File(log_path, 'r')
                    
                    #load cam1_ptr for alignment - this is essential
                    if 'cam1_ptr' in self.log_file:
                        self.cam1_ptr = self.log_file['cam1_ptr'][:]
                    else:
                        messagebox.showwarning("Warning", "Log file doesn't contain cam1_ptr data")
                        self.cam1_ptr = None
                    
                    #load all available metrics from our predefined list
                    for metric in self.metrics_list:
                        if metric in self.log_file:
                            self.metrics_data[metric] = self.log_file[metric][:]
                        else:
                            self.metrics_data[metric] = None
                    
                    #Enable toggle buttons for available metrics
                    available_metrics = [m for m in self.metrics_list if m in self.metrics_data and self.metrics_data[m] is not None]
                    for metric in self.metrics_list:
                        if metric in available_metrics:
                            self.toggle_buttons[metric].configure(state="normal")
                        else:
                            self.toggle_buttons[metric].configure(state="disabled")
                            
                    #Update metrics display based on currently toggled metrics
                    self.update_metrics_display()
                            
                    print(f"Loaded {len(available_metrics)} metrics: {', '.join(available_metrics)}")
                            
                except Exception as e:
                    messagebox.showwarning("Warning", f"Error opening log file: {str(e)}")
                    self.log_file = None
                    self.cam1_ptr = None
                    self.metrics_data = {}
            else:
                messagebox.showwarning("Warning", "Log file not found, continuing without metrics data")
                self.log_file = None
                self.cam1_ptr = None
                self.metrics_data = {}
            
            #update interface
            self.current_frame_idx = 0
            self.update_info(filename, camera_path)
            
            #update slider - do this before displaying frame to avoid recursive loop
            self.updating_slider = True
            self.frame_slider.config(to=self.total_frames-1)
            self.frame_slider.set(0)
            self.frame_slider.state(['!disabled'])
            self.updating_slider = False
            
            #now display the frame
            self.display_frame(self.current_frame_idx)
            
            #enable buttons
            self.play_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
    
    def update_metrics_display(self):
        #Create metrics display for toggled metrics
        self.create_metrics_display()
        
        #Update metric values if we have data
        if self.frames is not None:
            aligned_metrics = self.get_aligned_metrics(self.current_frame_idx)
            for metric, frame_info in self.metric_frames.items():
                if metric in aligned_metrics:
                    value = aligned_metrics[metric]
                    display_value = self.format_raw_value(value)
                    frame_info["value"].config(text=display_value)
    
    def get_aligned_metrics(self, frame_idx):
        #returns aligned metrics for the specified frame index
        if self.cam1_ptr is None:
            return {}
            
        #find indices where cam1_ptr equals the frame index
        matching_indices = np.where(self.cam1_ptr == frame_idx)[0]
        
        if len(matching_indices) == 0:
            return {}
            
        #get metrics for these indices
        aligned_metrics = {}
        for metric, data in self.metrics_data.items():
            if data is not None:
                try:
                    if len(data.shape) == 1:  #handle 1D arrays
                        #get average value for this frame
                        values = data[matching_indices]
                        if len(values) > 0:
                            aligned_metrics[metric] = np.mean(values)
                    else:  #handle multi-dimensional data
                        #just use the first matching index for simplicity
                        aligned_metrics[metric] = data[matching_indices[0]]
                except Exception as e:
                    print(f"Error processing metric {metric}: {str(e)}")
            
        return aligned_metrics
    
    def format_raw_value(self, value):
        # For arrays, show a summary to avoid taking too much space
        if isinstance(value, (np.ndarray, list)):
            if len(value) > 3:
                return f"{value[:3]}... (shape: {np.shape(value)})"
            return str(value)
        
        # For scalars, just convert to string with minimal formatting
        if isinstance(value, (int, np.integer)):
            return str(value)
        elif isinstance(value, (float, np.floating)):
            # Limit to 4 decimal places for float values for readability
            return f"{value:.4f}"
        else:
            return str(value)
    
    def update_info(self, filename, filepath):
        #update file info
        self.filename_label.config(text=f"File: {filename}")
        
        #calculate file size in GB
        file_size_bytes = os.path.getsize(filepath)
        file_size_gb = file_size_bytes / (1024**3)
        self.filesize_label.config(text=f"Size: {file_size_gb:.2f} GB")
        
        #update frame counter
        self.frame_label.config(text=f"Frame: {self.current_frame_idx + 1}/{self.total_frames}")
    
    def display_frame(self, idx):
        if self.frames is None or idx >= self.total_frames:
            return
        
        #get frame and transpose to (height, width, channels)
        frame = self.frames[idx][:].transpose(1, 2, 0)
        
        #convert to PIL Image
        img = Image.fromarray(np.uint8(frame))
        
        #resize to fit display area
        width, height = self.video_frame.winfo_width(), self.video_frame.winfo_height()
        if width > 1 and height > 1:  #ensure valid dimensions
            img = img.resize((width, height), Image.LANCZOS)
        
        #convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.photo)
        
        #update metrics display
        aligned_metrics = self.get_aligned_metrics(idx)
        for metric, frame_info in self.metric_frames.items():
            if metric in aligned_metrics:
                value = aligned_metrics[metric]
                display_value = self.format_raw_value(value)
                frame_info["value"].config(text=display_value)
            else:
                frame_info["value"].config(text="N/A")
        
        #update frame counter
        self.frame_label.config(text=f"Frame: {idx + 1}/{self.total_frames}")
        
        #update slider without triggering the callback
        if not self.updating_slider:
            self.updating_slider = True
            try:
                self.frame_slider.set(idx)
            finally:
                self.updating_slider = False
    
    def on_slider_change(self, value):
        #prevent recursive calls
        if self.updating_slider:
            return
            
        if self.frames is not None:
            #convert to integer and bounds check
            idx = min(max(0, int(float(value))), self.total_frames - 1)
            self.current_frame_idx = idx
            self.display_frame(idx)
    
    def play_video(self):
        if self.is_playing or self.frames is None:
            return
        
        self.is_playing = True
        
        #update button states
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        
        #create and start play thread
        self.play_thread = threading.Thread(target=self.play_loop)
        self.play_thread.daemon = True
        self.play_thread.start()
    
    def play_loop(self):
        frame_delay = 1/self.fps  #get current fps setting
        
        while self.is_playing and self.current_frame_idx < self.total_frames - 1:
            start_time = time.time()
            
            #increment frame index
            self.current_frame_idx += 1
            
            #use the main thread to update UI
            self.root.after(0, lambda idx=self.current_frame_idx: self.display_frame(idx))
            
            #calculate time to sleep
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
        
        #if reached the end, change state
        if self.current_frame_idx >= self.total_frames - 1:
            self.is_playing = False
            self.root.after(0, self.update_play_button)
    
    def update_play_button(self):
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
    
    def pause_video(self):
        if not self.is_playing:
            return
            
        self.is_playing = False
        
        #update button states
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        
        #wait for the play thread to end
        if self.play_thread is not None:
            self.play_thread.join(timeout=1.0)
    
    def reset_video(self):
        #pause if playing
        self.pause_video()
        
        #reset frame index
        self.current_frame_idx = 0
        self.display_frame(self.current_frame_idx)
    
    def on_closing(self):
        #close h5 files
        if self.camera_file is not None:
            self.camera_file.close()
        if self.log_file is not None:
            self.log_file.close()
        
        self.root.destroy()

def main():
    root = tk.Tk()
    app = H5VideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()