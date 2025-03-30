import tkinter as tk
import math
from fractions import Fraction

class SteeringWheelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Australian Steering Wheel Simulator")
        
        #set up canvas
        self.canvas_width = 600
        self.canvas_height = 600
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='#f0f0f0')
        self.canvas.pack(pady=20)
        
        #steering wheel parameters
        self.wheel_center_x = self.canvas_width // 2
        self.wheel_center_y = self.canvas_height // 2
        self.wheel_radius = 150
        self.current_angle = 0 #by default
        self.max_angle = 720 #2 full rotations anti-clock-wise
        self.min_angle = -720 #2 full rotations clock-wise
        
        #create the angle display frame
        self.angle_frame = tk.LabelFrame(root, text="Steering Angle", font=("Arial", 12, "bold"), padx=10, pady=10)
        self.angle_frame.pack(pady=10)
        
        self.angle_label_degrees = tk.Label(self.angle_frame, text="Angle (degrees): 0°", font=("Arial", 12))
        self.angle_label_degrees.pack(padx=10, pady=5)
        
        self.angle_label_radians = tk.Label(self.angle_frame, text="Angle (radians): 0", font=("Arial", 12))
        self.angle_label_radians.pack(padx=10, pady=5)

        self.reset_button = tk.Button(root, text="Reset Wheel", command=self.reset_wheel, font=("Arial", 10, "bold"))
        self.reset_button.pack(pady=10)
        
        #draw the steering wheel
        self.draw_steering_wheel()
        
        #bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        self.dragging = False
        self.last_x = 0
        self.last_y = 0

    #reset wheel to initial position
    def reset_wheel(self):
        #reset angle to initial position
        self.current_angle = 0
        
        #update display and redraw
        self.update_angle_display()
        self.draw_steering_wheel()

    
    def draw_steering_wheel(self):
        #clear canvas
        self.canvas.delete("wheel")
        
        #draw outer rim
        self.canvas.create_oval(
            self.wheel_center_x - self.wheel_radius,
            self.wheel_center_y - self.wheel_radius,
            self.wheel_center_x + self.wheel_radius,
            self.wheel_center_y + self.wheel_radius,
            width=8,
            outline="#333333",
            tags="wheel"
        )
        
        #draw inner circle (center hub)
        hub_radius = self.wheel_radius // 3
        self.canvas.create_oval(
            self.wheel_center_x - hub_radius,
            self.wheel_center_y - hub_radius,
            self.wheel_center_x + hub_radius,
            self.wheel_center_y + hub_radius,
            fill="#444444",
            outline="#222222",
            width=2,
            tags="wheel"
        )
        
        #extra shading inside inner circle
        logo_radius = hub_radius // 2
        self.canvas.create_oval(
            self.wheel_center_x - logo_radius,
            self.wheel_center_y - logo_radius,
            self.wheel_center_x + logo_radius,
            self.wheel_center_y + logo_radius,
            fill="#cccccc",
            outline="#999999",
            width=1,
            tags="wheel"
        )
        
        
        #draw spokes
        spoke_length = self.wheel_radius - hub_radius
        spoke_width = 20
        
        #calculate spoke positions based on current angle
        angle_rad = math.radians(self.current_angle)
        
        #draw three spokes (120 degrees apart)
        for i in range(3):
            #seperate 3 spokes by 120 degrees
            spoke_angle = angle_rad + i * (2 * math.pi / 3)
            spoke_end_x = self.wheel_center_x + spoke_length * math.cos(spoke_angle)
            spoke_end_y = self.wheel_center_y + spoke_length * math.sin(spoke_angle)
            
            self.canvas.create_line(
                self.wheel_center_x + hub_radius * math.cos(spoke_angle),
                self.wheel_center_y + hub_radius * math.sin(spoke_angle),
                spoke_end_x,
                spoke_end_y,
                width=spoke_width,
                fill="#333333",
                tags="wheel"
            )
    
    def on_mouse_press(self, event):
        #check if click is within the steering wheel
        #calculate distance vector of mouse click position to wheel's centre
        dx = event.x - self.wheel_center_x
        dy = event.y - self.wheel_center_y
        distance = math.sqrt(dx**2 + dy**2)
        
        #if the distance or if the mouse is clicked within the wheel's radius then move
        if distance <= self.wheel_radius:
            self.dragging = True
            self.last_x = event.x
            self.last_y = event.y
    
    #when mouse has clicked on wheel and is dragging it
    def on_mouse_drag(self, event):
        if not self.dragging:
            return
        
        #calculate the angle change based on mouse movement
        dx1 = self.last_x - self.wheel_center_x
        dy1 = self.last_y - self.wheel_center_y
        dx2 = event.x - self.wheel_center_x
        dy2 = event.y - self.wheel_center_y
        
        #calculate the angle between the two vectors (previous click and current click position)
        #math atan2 used to calculate clockwise angle of x and y in radians
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        
        #calculate the change in angle
        delta_angle = math.degrees(angle2 - angle1)
        
        #adjust for angle wrapping
        if delta_angle > 180:
            delta_angle -= 360
        elif delta_angle < -180:
            delta_angle += 360
        
        #update the current angle (with limits)
        new_angle = self.current_angle + delta_angle
        
        #enforce rotation limits (+-720 degrees)
        if new_angle > self.max_angle:
            new_angle = self.max_angle
        elif new_angle < self.min_angle:
            new_angle = self.min_angle
        
        self.current_angle = new_angle
        
        #update the last position
        self.last_x = event.x
        self.last_y = event.y
        
        #redraw the steering wheel and update angle display
        self.update_angle_display()
        self.draw_steering_wheel()
    
    #when mouse is released
    def on_mouse_release(self, event):
        self.dragging = False
    
    #updating angle display (degrees and radians)
    def update_angle_display(self):
        #get the angles
        degrees = round(self.current_angle)
        radians_decimal = round(math.radians(self.current_angle), 2)
        
        #calculate how many π's we have
        pi_multiples = self.current_angle / 180
        
        #handle the sign
        sign = ""
        if pi_multiples < 0:
            sign = "-"
            pi_multiples = abs(pi_multiples)
        
        #format the π representation using fractions
        frac = Fraction(pi_multiples).limit_denominator(100)
        
        if frac.numerator == 0:
            radians_text = "0"
        elif frac.numerator == 1:
            if frac.denominator == 1:
                radians_text = f"{sign}π"
            else:
                radians_text = f"{sign}π/{frac.denominator}"
        else:
            if frac.denominator == 1:
                radians_text = f"{sign}{frac.numerator}π"
            else:
                radians_text = f"{sign}{frac.numerator}π/{frac.denominator}"
        
        #also show the decimal value
        radians_text += f" ({radians_decimal})"
        
        self.angle_label_degrees.config(text=f"Angle (degrees): {degrees}°")
        self.angle_label_radians.config(text=f"Angle (radians): {radians_text}")

def main():
    root = tk.Tk()
    app = SteeringWheelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()