import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import io

#class to manage and render a speed graph with gradient coloring
class SpeedGraph:
    def __init__(self, max_points=24, max_speed=30, fps=20):
        #set up parameters
        self.max_points = max_points  #maximum number of points to display
        self.max_speed = max_speed  #maximum y-axis value
        self.fps = fps  #frames per second for time step calculation
        
        #display configuration
        self.mode = "speed"  #display mode: "speed" or "acceleration"
        self.fixed_ylim = True  #toggle between fixed or dynamic y-axis limits
        
        #tracking variables for dynamic y-axis
        self.min_observed = float('inf')
        self.max_observed = float('-inf')
        self.zoom_enabled = False  #only enable zoom after collecting enough data
        
        #gradient colors for line segments (true speed - red-orange to purple)
        self.true_colors = [
            '#FF9980', '#FF8E8A', '#FF8394', '#FF789E', '#FF6DA8', '#FF62B2', '#FF57BC', '#FF4CC6',
            '#F941D0', '#F236DA', '#EA2BE4', '#E220EE', '#D916F8', '#CF0DFF', '#C40DFF', '#BA0DFF',
            '#B00DFF', '#A60DFF', '#9C0DFF', '#920DFF', '#880DFF', '#7E0DFF', '#740DFF', '#6A0DFF'
        ]
        
        #gradient colors for simulated speed (cyan to white)
        self.sim_colors = [
            '#FFFFFF', '#F0FFFF', '#E0FFFF', '#D0FFFF', '#C0FFFF', 
            '#B0FFFF', '#A0FFFF', '#90FFFF', '#80FFF8', '#70FFF0', 
            '#60FFE8', '#50FFE0', '#40FFD8', '#30FFD0', '#20FFC8', 
            '#10FFC0', '#00FFB8', '#00F0B0', '#00E8A8', '#00E0A0', 
            '#00D898', '#00D090', '#00C888', '#00C080'
        ]
        
        #initialise data buffers for true speed
        self.true_speeds = []  #buffer for true speed values
        self.frames = []  #buffer for corresponding frame indices
        self.last_frame = -1  #track last frame to avoid redundant updates
        
        #initialise data buffers for simulated speed
        self.sim_speeds = []  #buffer for simulated speed values
        self.show_sim_speed = False  #flag to control display of simulated speed
        self.current_sim_speed = 0.0  #current simulated speed
        
        #create figure and axes with fixed size and matching background color
        self.fig, self.ax = plt.subplots(figsize=(8, 5), facecolor='#2C2C30', dpi=100)
        self.setup_plot()
    
    def setup_plot(self):
        #configure plot appearance
        self.ax.set_facecolor('#2C2C30')
        
        if self.mode == "speed":
            #set y-axis limits for speed mode - fixed or dynamic
            if not self.fixed_ylim and self.zoom_enabled and self.min_observed < float('inf') and self.max_observed > float('-inf'):
                #use zoomed limits with 0.1 buffer
                buffer = 0.1
                y_min = max(0, self.min_observed - buffer)
                y_max = self.max_observed + buffer
                self.ax.set_ylim(y_min, y_max)
            else:
                #use default limits for speed mode
                self.ax.set_ylim(0, self.max_speed)
            
            #title for speed mode
            self.ax.set_title('Speed (m/s)', fontsize=12, color='white')
        else:  #acceleration mode
            #always use fixed limits for acceleration mode (-1 to 1)
            self.ax.set_ylim(-5, 5)
            
            #title for acceleration mode
            self.ax.set_title('Acceleration (m/s²)', fontsize=12, color='white')
        
        #grid settings
        self.ax.grid(True, linestyle='-', alpha=0.7, color='#555555')
        
        #customise spines
        for spine in self.ax.spines.values():
            spine.set_color('#555555')
        
        #text colors
        self.ax.tick_params(colors='white')
        self.ax.yaxis.label.set_color('white')
        self.ax.xaxis.label.set_color('white')
        self.ax.title.set_color('white')
        
        #labels
        self.ax.set_xlabel('Frame', fontsize=10, color='white')
        
        #ensure tight layout
        self.fig.tight_layout()
    
    def calculate_simulated_speed(self, true_speed, acceleration, model_running):
        #calculate simulated speed based on model's acceleration applied to true speed
        time_step = 1.0 / self.fps
        
        #handle case when model is not running
        if not model_running:
            return true_speed
        
        #apply acceleration to true speed for one time step
        sim_speed = true_speed + (acceleration * time_step)
        
        #ensure speed doesn't go negative
        sim_speed = max(0.0, sim_speed)
        
        return sim_speed
        
    def update(self, frame_idx, true_value, sim_value=None, model_running=False):
        #set model running state
        self.show_sim_speed = model_running
        
        #handle slider going backward by clearing points after current frame
        if frame_idx < self.last_frame:
            #find position to trim from
            trim_pos = None
            for i, f in enumerate(self.frames):
                if f > frame_idx:
                    trim_pos = i
                    break
            
            #trim data if position found
            if trim_pos is not None:
                self.frames = self.frames[:trim_pos]
                self.true_speeds = self.true_speeds[:trim_pos]
                self.sim_speeds = self.sim_speeds[:trim_pos]
        
        #skip update if this is the same frame
        if frame_idx == self.last_frame and len(self.frames) > 0:
            return self.get_image()
        
        self.last_frame = frame_idx
        
        #add true value point
        self.frames.append(frame_idx)
        self.true_speeds.append(true_value)
        
        #add simulated value if model is running
        if model_running and sim_value is not None:
            self.sim_speeds.append(sim_value)
            
            #update min/max observed values for dynamic y-axis in speed mode
            if self.mode == "speed" and not self.fixed_ylim:
                self.min_observed = min(self.min_observed, true_value, sim_value)
                self.max_observed = max(self.max_observed, true_value, sim_value)
                
                #enable zoom after collecting enough points
                if len(self.frames) >= 3:
                    self.zoom_enabled = True
        else:
            #use true value as placeholder (won't be displayed when model not running)
            self.sim_speeds.append(true_value)
            
            #update min/max for true value only in speed mode
            if self.mode == "speed" and not self.fixed_ylim:
                self.min_observed = min(self.min_observed, true_value)
                self.max_observed = max(self.max_observed, true_value)
                
                #enable zoom after collecting enough points
                if len(self.frames) >= 3:
                    self.zoom_enabled = True
        
        #maintain max buffer size by removing oldest points if needed
        if len(self.frames) > self.max_points:
            self.frames.pop(0)
            self.true_speeds.pop(0)
            self.sim_speeds.pop(0)
            
            #recalculate min/max after removing oldest points in speed mode
            if self.mode == "speed" and not self.fixed_ylim and len(self.frames) > 0:
                self.min_observed = min(min(self.true_speeds), min(self.sim_speeds))
                self.max_observed = max(max(self.true_speeds), max(self.sim_speeds))
        
        #clear previous plot
        self.ax.clear()
        
        #re-setup plot aesthetics
        self.setup_plot()
        
        #plot true speed with gradient colors if we have at least 2 points
        if len(self.true_speeds) >= 2:
            for i in range(len(self.true_speeds) - 1):
                #use gradient colors based on position in the sequence
                color_idx = min(i, len(self.true_colors) - 1)
                self.ax.plot(
                    self.frames[i:i+2], 
                    self.true_speeds[i:i+2], 
                    color=self.true_colors[color_idx], 
                    linewidth=2
                )
        
        #plot simulated speed with gradient colors if enabled and we have at least 2 points
        if self.show_sim_speed and len(self.sim_speeds) >= 2:
            for i in range(len(self.sim_speeds) - 1):
                #use gradient colors based on position in the sequence
                color_idx = min(i, len(self.sim_colors) - 1)
                self.ax.plot(
                    self.frames[i:i+2], 
                    self.sim_speeds[i:i+2], 
                    color=self.sim_colors[color_idx], 
                    linewidth=2
                )
        
        #return rendered image
        return self.get_image()
    
    def get_image(self):
        #render matplotlib figure to a PIL image
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', facecolor=self.fig.get_facecolor(), dpi=100)
        buf.seek(0)
        
        img = Image.open(buf)
        return img
    
    def reset(self):
        #clear data when loading a new file
        self.frames = []
        self.true_speeds = []
        self.sim_speeds = []
        self.last_frame = -1
        self.show_sim_speed = False
        self.current_sim_speed = 0.0
        
        #reset zoom tracking
        self.min_observed = float('inf')
        self.max_observed = float('-inf')
        self.zoom_enabled = False
        
        self.ax.clear()
        self.setup_plot()
        return self.get_image()
        
    def cleanup(self):
        #simple cleanup - close the figure and clear references
        if hasattr(self, 'fig') and self.fig is not None:
            #close the figure window
            plt.close(self.fig)
            
            #clear references
            self.fig = None
            self.ax = None