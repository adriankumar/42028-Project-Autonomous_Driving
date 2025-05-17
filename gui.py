from gui.gui_main import VideoGui
import tkinter as tk

#run this script to open GUI 
if __name__=='__main__':
    root = tk.Tk()
    root.geometry('1051x584')
    app = VideoGui(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()