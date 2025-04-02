import pygame
import math
import os

class SteeringWheelSimulator:
    def __init__(self):
        #initialise pygame
        pygame.init()
        
        #window dimensions
        self.width = 600
        self.height = 650  #extra height for angle display
        
        #set up display
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pygame Steering Wheel Simulator")
        
        #colours
        self.bg_color = (240, 240, 240)
        self.wheel_color = (51, 51, 51)
        self.hub_color = (68, 68, 68)
        self.hub_logo_color = (204, 204, 204)
        self.text_color = (0, 0, 0)
        
        #wheel parameters
        self.wheel_center = (self.width // 2, self.height // 2 - 25)  #offset a bit for text display
        self.wheel_radius = 150
        self.hub_radius = self.wheel_radius // 3
        self.logo_radius = self.hub_radius // 2
        self.spoke_width = 20
        
        #angle parameters
        self.current_angle = 0.0
        self.max_angle = 720  #2 full rotations clockwise
        self.min_angle = -720  #2 full rotations anti-clockwise
        
        #interaction states
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        
        #file output
        self.last_written_angle = None
        
        #font initialisation
        self.font = pygame.font.SysFont('Arial', 24)
        self.reset_font = pygame.font.SysFont('Arial', 16, bold=True)
        
        #for reset button
        self.reset_button = pygame.Rect(self.width // 2 - 60, self.height - 50, 120, 30)
        
        #create directory for angle output
        os.makedirs("./testing_stuff/angle_output", exist_ok=True)
    
    def draw_steering_wheel(self):
        #draw outer rim
        pygame.draw.circle(self.screen, self.wheel_color, self.wheel_center, self.wheel_radius, 8)
        
        #draw hub (centre)
        pygame.draw.circle(self.screen, self.hub_color, self.wheel_center, self.hub_radius)
        pygame.draw.circle(self.screen, self.hub_logo_color, self.wheel_center, self.logo_radius)
        
        #draw spokes
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
            pygame.draw.line(self.screen, self.wheel_color, spoke_start, spoke_end, self.spoke_width)
    
    def draw_angle_display(self):
        #create and display the angle text
        angle_text = f"Angle (degrees): {self.current_angle:.2f}°"
        text_surface = self.font.render(angle_text, True, self.text_color)
        text_rect = text_surface.get_rect(center=(self.width // 2, self.height - 90))
        self.screen.blit(text_surface, text_rect)
        
        #draw reset button
        pygame.draw.rect(self.screen, (180, 180, 180), self.reset_button, border_radius=5)
        reset_text = self.reset_font.render("Reset Wheel", True, (0, 0, 0))
        reset_text_rect = reset_text.get_rect(center=self.reset_button.center)
        self.screen.blit(reset_text, reset_text_rect)
    
    def is_mouse_on_wheel(self, mouse_pos):
        dx = mouse_pos[0] - self.wheel_center[0]
        dy = mouse_pos[1] - self.wheel_center[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance <= self.wheel_radius
    
    def calculate_angle_change(self, new_pos):
        #calculate vectors from centre to old and new positions
        old_vector = (self.last_mouse_pos[0] - self.wheel_center[0], 
                     self.last_mouse_pos[1] - self.wheel_center[1])
        new_vector = (new_pos[0] - self.wheel_center[0], 
                     new_pos[1] - self.wheel_center[1])
        
        #calculate angles of both vectors
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
        
        #update angle with limits
        new_angle = self.current_angle + delta_angle
        if new_angle > self.max_angle:
            new_angle = self.max_angle
        elif new_angle < self.min_angle:
            new_angle = self.min_angle
            
        self.current_angle = new_angle
        self.last_mouse_pos = mouse_pos
        
        #write angle to file if it changed
        self.write_angle_to_file()
    
    # def write_angle_to_file(self):
    #     #write the angle to file if it's different from last written value
    #     if self.last_written_angle is None or self.last_written_angle != self.current_angle:
    #         with open("./testing_stuff/angle_output/steering_angle.txt", "w") as f:
    #             f.write(f"{self.current_angle:.2f}")
    #         self.last_written_angle = self.current_angle

    def write_angle_to_file(self):
        #write the angle on every frame update, regardless of whether it changed
        with open("./testing_stuff/angle_output/steering_angle.txt", "w") as f:
            f.write(f"{self.current_angle:.2f}")
        self.last_written_angle = self.current_angle
    
    def reset_wheel(self):
        self.current_angle = 0.0
        self.write_angle_to_file()
    
    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                #mouse button down
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  #left mouse button
                        mouse_pos = pygame.mouse.get_pos()
                        if self.is_mouse_on_wheel(mouse_pos):
                            self.dragging = True
                            self.last_mouse_pos = mouse_pos
                        elif self.reset_button.collidepoint(mouse_pos):
                            self.reset_wheel()
                
                #mouse button up
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                
                #mouse motion while dragging
                elif event.type == pygame.MOUSEMOTION and self.dragging:
                    self.update_wheel_angle(pygame.mouse.get_pos())
            
            #fill background
            self.screen.fill(self.bg_color)
            
            #draw steering wheel and angle display
            self.draw_steering_wheel()
            self.draw_angle_display()
            
            #update display
            pygame.display.flip()
            
            #cap the frame rate
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    simulator = SteeringWheelSimulator()
    simulator.run()