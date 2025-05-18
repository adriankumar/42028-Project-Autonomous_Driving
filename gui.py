# from gui_alt.gui_main import VideoGui
from gui_efficient.gui_main import VideoGui
# from gui.gui_main import VideoGui #uncomment this and comment the above import to utilise the initial GUI application
import tkinter as tk


#run this script to open GUI 
if __name__=='__main__':
    root = tk.Tk()
    root.geometry('1051x584')
    app = VideoGui(root)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()