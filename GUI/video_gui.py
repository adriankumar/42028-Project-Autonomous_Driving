import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import h5py
import numpy as np
import os
from PIL import Image, ImageTk
import time
import threading
import steering_trajectory_calculator as stc

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
        
        #telemetry categories
        self.telemetry_categories = {
            "Primary Control": ["steering_angle", "speed"],
            "Secondary Control": ["steering_torque", "speed_abs", "car_accel"],
            "Context Indicators": ["blinker", "standstill"]
        }
        
        #telemetry units
        self.telemetry_units = {
            "steering_angle": "deg",
            "speed": "m/s",
            "steering_torque": "deg/s",
            "speed_abs": "m/s",
            "car_accel": "m/s²",
            "blinker": "bool",
            "standstill": "bool"
        }
        
        #telemetry data storage
        self.telemetry_data = {}
        self.cam1_ptr = None
        
        #create UI components
        self.create_frames()
        self.create_widgets()
        
        #bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)
        
        #print dictionary keys for debugging
        print(f"Info label keys: {list(self.info_labels.keys())}")

    def create_frames(self):
        #main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        #left and right frames (main split)
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        #fixed width for the right information panel to ensure all content is visible
        self.right_frame = ttk.Frame(self.main_frame, width=300)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.right_frame.pack_propagate(False)  #prevent frame from shrinking
        
        #video controls, scrollbar, and display frames (left side)
        self.video_buttons_frame = ttk.Frame(self.left_frame)
        self.video_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.scrollbar_frame = ttk.Frame(self.left_frame)
        self.scrollbar_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.video_display_frame = ttk.Frame(self.left_frame, borderwidth=2, relief=tk.GROOVE)
        self.video_display_frame.pack(fill=tk.BOTH, expand=True)
        
        #information and telemetry frames (right side)
        self.information_frame = ttk.LabelFrame(self.right_frame, text="Information")
        self.information_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.toggle_data_frame = ttk.LabelFrame(self.right_frame, text="Telemetry Type")
        self.toggle_data_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.visualisation_choice_frame = ttk.LabelFrame(self.right_frame, text="Visualisation")
        self.visualisation_choice_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.visualisation_frame = ttk.LabelFrame(self.right_frame, text="Telemetry Data")
        self.visualisation_frame.pack(fill=tk.BOTH, expand=True)

    def create_widgets(self):
        #create information display first so info_labels is initialised
        self.create_info_display()
        
        #create video control buttons
        self.create_video_controls()
        
        #create scrollbar
        self.create_scrollbar()
        
        #create video display
        self.create_video_display()
        
        #create telemetry type selector
        self.create_telemetry_selector()
        
        #create visualisation options
        self.create_visualization_options()
        
        #create empty telemetry display
        self.telemetry_frames = {}
        self.telemetry_values = {}

    def create_video_controls(self):
        #create buttons with a common function
        self.load_btn = self.create_button(
            self.video_buttons_frame, "Load H5 File", self.load_h5_file)
        self.play_btn = self.create_button(
            self.video_buttons_frame, "Play", self.play_video, state=tk.DISABLED)
        self.pause_btn = self.create_button(
            self.video_buttons_frame, "Pause", self.pause_video, state=tk.DISABLED)
        self.reset_btn = self.create_button(
            self.video_buttons_frame, "Reset", self.reset_video, state=tk.DISABLED)
        
        #fps control
        fps_frame = ttk.Frame(self.video_buttons_frame)
        fps_frame.pack(side=tk.RIGHT)
        ttk.Label(fps_frame, text="FPS:").pack(side=tk.LEFT)
        self.fps_var = tk.StringVar(value="20")
        fps_spinbox = ttk.Spinbox(fps_frame, from_=1, to=60, width=3, 
                                 textvariable=self.fps_var, command=self.update_fps)
        fps_spinbox.pack(side=tk.LEFT)

    def create_button(self, parent, text, command, **kwargs):
        btn = ttk.Button(parent, text=text, command=command, **kwargs)
        btn.pack(side=tk.LEFT, padx=2)
        return btn

    def create_scrollbar(self):
        self.frame_slider = ttk.Scale(
            self.scrollbar_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
            command=self.on_slider_change)
        self.frame_slider.pack(fill=tk.X)
        self.frame_slider.state(['disabled'])

    def create_video_display(self):
        #video display (left side) - set width/height based on 320x160 aspect ratio
        self.video_frame = ttk.Frame(self.video_display_frame, width=640, height=320)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_frame.pack_propagate(False)  #prevent frame from shrinking
        
        self.video_label = ttk.Label(self.video_frame, background="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        #annotation toggle (next to video)
        self.annotation_var = tk.BooleanVar(value=False)
        self.annotation_toggle = ttk.Checkbutton(
            self.video_display_frame, text="Show Steering Trajectory", 
            variable=self.annotation_var, command=self.toggle_annotations)
        
        self.annotation_toggle.pack(side=tk.RIGHT, padx=5)

    def create_info_display(self):
        #file information display (top right)
        labels = ["Name: ", "Size: ", "Frame: ", "Time: "]
        self.info_labels = {}
        
        for label in labels:
            frame = ttk.Frame(self.information_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=label, width=8).pack(side=tk.LEFT)
            # Use fixed width for value labels to prevent layout changes
            value_label = ttk.Label(frame, text="N/A", width=25, anchor=tk.W)
            value_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            key = label.strip(": ")  #remove both colon and space
            self.info_labels[key] = value_label

    def create_telemetry_selector(self):
        #telemetry type selector (middle right)
        self.telemetry_type_var = tk.StringVar()
        
        for i, category in enumerate(self.telemetry_categories.keys()):
            rb = ttk.Radiobutton(
                self.toggle_data_frame, text=category, value=category,
                variable=self.telemetry_type_var, command=self.update_telemetry_display)
            rb.pack(anchor=tk.W, pady=2)
            
            #select the first one by default
            if i == 0:
                self.telemetry_type_var.set(category)

    def create_visualization_options(self):
        #visualisation options (below telemetry selector)
        self.viz_choice_var = tk.StringVar(value="Value")
        
        viz_frame = ttk.Frame(self.visualisation_choice_frame)
        viz_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(
            viz_frame, text="Value", value="Value",
            variable=self.viz_choice_var, command=self.update_visualization).pack(side=tk.LEFT, padx=5)
        
        ttk.Radiobutton(
            viz_frame, text="Graph", value="Graph",
            variable=self.viz_choice_var, command=self.show_graph_not_implemented).pack(side=tk.LEFT, padx=5)

    def create_telemetry_frame(self, parent, name, prefix, index):
        #create a frame for a single telemetry data point
        frame_id = f"{prefix}_frame_{index}"
        frame = ttk.Frame(parent)
        
        #telemetry name
        name_label = ttk.Label(frame, text=name.replace("_", " ").title(), font=("Arial", 10, "bold"))
        name_label.pack(anchor=tk.W)
        
        #telemetry value with fixed width to prevent layout changes
        value_label = ttk.Label(frame, text="N/A", width=20)
        value_label.pack(anchor=tk.W, padx=10)
        
        #unit of measurement
        unit = self.telemetry_units.get(name, "")
        unit_label = ttk.Label(frame, text=f"({unit})" if unit else "")
        unit_label.pack(anchor=tk.W, padx=10)
        
        return frame, value_label

    def update_telemetry_display(self):
        #clear current telemetry frames
        for frame in self.telemetry_frames.values():
            frame.pack_forget()
        
        self.telemetry_frames = {}
        self.telemetry_values = {}
        
        #get selected category
        category = self.telemetry_type_var.get()
        if not category:
            return
            
        #get telemetry items for this category
        telemetry_items = self.telemetry_categories.get(category, [])
        
        #create frames for each telemetry item
        for i, item in enumerate(telemetry_items):
            prefix = ''.join([word[0] for word in category.split()])  #e.g., "Primary Control" -> "PC"
            frame, value_label = self.create_telemetry_frame(
                self.visualisation_frame, item, prefix, i+1)
            
            frame.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=5, pady=5)
            
            self.telemetry_frames[item] = frame
            self.telemetry_values[item] = value_label
        
        #update values if data is loaded
        if self.frames is not None:
            self.update_telemetry_values()

    def update_telemetry_values(self):
        #update telemetry values for current frame
        aligned_data = self.get_aligned_data(self.current_frame_idx)
        
        for item, label in self.telemetry_values.items():
            if item in aligned_data:
                value = aligned_data[item]

                if item == "steering_angle":
                    value = value / 10.0

                #Format numeric values to 2 decimal places
                if isinstance(value, (float, np.float32, np.float64)):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                label.config(text=formatted_value)
            else:
                label.config(text="N/A")

    def update_visualization(self):
        #update based on selected visualisation type
        viz_type = self.viz_choice_var.get()
        if viz_type == "Value":
            self.update_telemetry_display()

    def show_graph_not_implemented(self):
        #show error that graph visualisation is not implemented
        self.show_error_message("Graph visualisation not implemented yet")
        self.viz_choice_var.set("Value")

    def toggle_annotations(self):
        #handle annotation toggle - now displays trajectory prediction
        if self.annotation_var.get():
            #annotations enabled, redisplay current frame with annotations
            self.display_frame(self.current_frame_idx)
        else:
            #annotations disabled, redisplay current frame without annotations
            self.display_frame(self.current_frame_idx)

    def show_error_message(self, message):
        #generic error message function
        messagebox.showinfo("Information", message)

    def load_h5_file(self):
        #stop any ongoing playback
        self.pause_video()
        
        #open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Camera H5 File",
            filetypes=[("HDF5 Files", "*.h5"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return  #user cancelled
            
        try:
            #close any open files
            if self.camera_file is not None:
                self.camera_file.close()
                self.camera_file = None
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
                
            #reset data
            self.telemetry_data = {}
            self.cam1_ptr = None
            
            #get filename and directory
            filename = os.path.basename(file_path)
            directory = os.path.dirname(file_path)
            
            #check if it's a camera file
            if "camera" in directory.lower():
                camera_path = file_path
                log_path = file_path.replace("camera", "log") #assumes dataset folder set up has 'logs' folder in it too
            else:
                #try to find associated log file
                camera_path = file_path
                log_path = os.path.join(
                    os.path.dirname(os.path.dirname(file_path)),
                    "log",
                    os.path.basename(file_path)
                )
            
            #load camera file
            self.load_camera_file(camera_path)
            
            #load log file if available
            if os.path.exists(log_path):
                self.load_log_file(log_path)
            else:
                self.show_error_message(f"Log file not found at {log_path}")
            
            #update UI
            self.update_ui_after_loading(filename, camera_path)
            
        except Exception as e:
            self.show_error_message(f"Error loading file: {str(e)}")

    def load_camera_file(self, file_path):
        #load camera H5 file
        try:
            self.camera_file = h5py.File(file_path, 'r')
            
            if 'X' not in self.camera_file:
                self.show_error_message("Camera file doesn't contain 'X' dataset")
                self.camera_file.close()
                self.camera_file = None
                return False
                
            self.frames = self.camera_file['X']
            self.total_frames = self.frames.shape[0]
            return True
            
        except Exception as e:
            self.show_error_message(f"Error opening camera file: {str(e)}")
            return False

    def load_log_file(self, file_path):
        #load log H5 file
        try:
            self.log_file = h5py.File(file_path, 'r')
            
            #load cam1_ptr for frame alignment
            if 'cam1_ptr' in self.log_file:
                self.cam1_ptr = self.log_file['cam1_ptr'][:]
            else:
                self.show_error_message("Log file doesn't contain 'cam1_ptr' dataset")
                
            #load telemetry data
            for category, items in self.telemetry_categories.items():
                for item in items:
                    if item in self.log_file:
                        self.telemetry_data[item] = self.log_file[item][:]
                    else:
                        self.telemetry_data[item] = None
            
            #load time data if available
            if 'times' in self.log_file:
                self.telemetry_data['times'] = self.log_file['times'][:]
                
            return True
            
        except Exception as e:
            self.show_error_message(f"Error opening log file: {str(e)}")
            return False

    def update_ui_after_loading(self, filename, file_path):
        #update UI after loading files
        self.current_frame_idx = 0
        
        #update file info
        file_size_bytes = os.path.getsize(file_path)
        file_size_gb = file_size_bytes / (1024**3)
        
        if "Name" in self.info_labels:
            self.info_labels["Name"].config(text=filename)
        if "Size" in self.info_labels:
            self.info_labels["Size"].config(text=f"{file_size_gb:.2f} GB")
        if "Frame" in self.info_labels:
            self.info_labels["Frame"].config(text=f"1/{self.total_frames}")
        
        #update time if available
        if 'times' in self.telemetry_data and self.telemetry_data['times'] is not None:
            time_data = self.get_aligned_data(0).get('times', 0)
            if "Time" in self.info_labels:
                self.info_labels["Time"].config(text=f"{time_data:.2f} s")
        
        #update slider
        self.updating_slider = True
        self.frame_slider.config(to=self.total_frames-1)
        self.frame_slider.set(0)
        self.frame_slider.state(['!disabled'])
        self.updating_slider = False
        
        #enable buttons
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.NORMAL)
        
        #display first frame
        self.display_frame(0)
        
        #update telemetry display
        self.update_telemetry_display()

    def get_aligned_data(self, frame_idx):
        #get telemetry data aligned with current frame
        if self.cam1_ptr is None or len(self.cam1_ptr) == 0:
            return {}
            
        #find indices where cam1_ptr equals the frame index
        matching_indices = np.where(self.cam1_ptr == frame_idx)[0]
        
        if len(matching_indices) == 0:
            return {}
            
        #get telemetry data for matching indices
        aligned_data = {}
        for key, data in self.telemetry_data.items():
            if data is not None and len(data) > 0:
                if len(matching_indices) > 0 and matching_indices[0] < len(data):
                    if len(data.shape) == 1:
                        #use first matching index for simplicity
                        aligned_data[key] = data[matching_indices[0]]
                    else:
                        #handle multi-dimensional data
                        aligned_data[key] = data[matching_indices[0]]
        
        return aligned_data

    def display_frame(self, idx):
        if self.frames is None or idx >= self.total_frames:
            return
            
        #get frame data and transpose to (height, width, channels)
        frame = self.frames[idx][:].transpose(1, 2, 0)

        #add annotations if enabled
        if self.annotation_var.get():
            #get aligned telemetry data for annotations
            aligned_data = self.get_aligned_data(idx)
            
            #only draw if we have steering angle and speed data
            if 'steering_angle' in aligned_data and 'speed' in aligned_data:
                true_angle = float(aligned_data['steering_angle'])
                speed = float(aligned_data['speed'])
                
                #draw annotations on frame using the trajectory calculator
                frame = stc.draw_trajectory_annotation(frame, speed, true_angle)
        
        #convert to PIL Image
        img = Image.fromarray(np.uint8(frame))
        
        #resize to fit display area (preserving aspect ratio)
        width = self.video_frame.winfo_width()
        height = self.video_frame.winfo_height()
        
        if width > 1 and height > 1:  #ensure valid dimensions
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height
            
            # Default aspect ratio is 320:160 = 2:1
            # Ensure we maintain this aspect ratio when resizing
            if width / height > aspect_ratio:
                new_width = int(height * aspect_ratio)
                new_height = height
            else:
                new_width = width
                new_height = int(width / aspect_ratio)
                
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        #convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=self.photo)
        
        #update frame counter
        if "Frame" in self.info_labels:
            self.info_labels["Frame"].config(text=f"{idx + 1}/{self.total_frames}")
        
        #update time if available
        aligned_data = self.get_aligned_data(idx)
        if 'times' in aligned_data and "Time" in self.info_labels:
            self.info_labels["Time"].config(text=f"{aligned_data['times']:.2f} s")
        
        #update telemetry values
        self.update_telemetry_values()
        
        #update slider without triggering callback
        if not self.updating_slider:
            self.updating_slider = True
            self.frame_slider.set(idx)
            self.updating_slider = False

    def on_slider_change(self, value):
        #handle slider change
        if self.updating_slider or self.frames is None:
            return
            
        idx = min(max(0, int(float(value))), self.total_frames - 1)
        self.current_frame_idx = idx
        self.display_frame(idx)

    def update_fps(self):
        #update fps value
        try:
            self.fps = int(self.fps_var.get())
        except ValueError:
            self.fps = 20
            self.fps_var.set("20")

    def play_video(self):
        if self.is_playing or self.frames is None:
            return
            
        self.is_playing = True
        
        #update button states
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        
        #start playback in a separate thread
        self.play_thread = threading.Thread(target=self.play_loop)
        self.play_thread.daemon = True
        self.play_thread.start()

    def play_loop(self):
        #playback loop
        frame_delay = 1.0 / self.fps
        
        while self.is_playing and self.current_frame_idx < self.total_frames - 1:
            start_time = time.time()
            
            #increment frame index
            self.current_frame_idx += 1
            
            #update UI in main thread
            self.root.after(0, lambda idx=self.current_frame_idx: self.display_frame(idx))
            
            #calculate delay
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_delay - elapsed)
            time.sleep(sleep_time)
        
        #update UI when playback ends
        if self.current_frame_idx >= self.total_frames - 1:
            self.is_playing = False
            self.root.after(0, self.update_play_button)

    def update_play_button(self):
        #update play button state after playback ends
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)

    def pause_video(self):
        if not self.is_playing:
            return
            
        self.is_playing = False
        
        #update button states
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        
        #wait for play thread to end
        if self.play_thread is not None:
            self.play_thread.join(timeout=0.5)

    def reset_video(self):
        #pause if playing
        self.pause_video()
        
        #reset frame index
        self.current_frame_idx = 0
        self.display_frame(self.current_frame_idx)

    def on_window_resize(self, event):
        #handle window resize
        if event.widget == self.root and self.frames is not None and hasattr(self, 'current_frame_idx'):
            #redisplay current frame to fit new size
            self.display_frame(self.current_frame_idx)

    def on_closing(self):
        #clean up when closing
        self.pause_video()
        
        if self.camera_file is not None:
            self.camera_file.close()
        if self.log_file is not None:
            self.log_file.close()
            
        self.root.destroy()

def main():
    root = tk.Tk()
    # Set default window size to better accommodate video dimensions (320x160) plus info panel
    root.geometry("1200x700")
    # Set minimum size to prevent UI elements from getting too compressed
    root.minsize(900, 500)
    app = H5VideoPlayer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()