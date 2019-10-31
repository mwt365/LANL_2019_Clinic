# backend libraries
import spectrogram
from spectrogram import Spectrogram
import numpy as np
import os

# all tkinter libraries
import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog

# all matplotlib libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
matplotlib.use("TkAgg")


# specify this path to the directory holding the dig files. 
PATH_TO_DIR = "/Users/trevorwalker/Desktop/Clinic/LANL_2019_Clinic/"

# customizable window-specific preferences
LARGE_FONT= ("Verdana", 12)
WINDOW_SIZE = "600x500"
WINDOW_TITLE = "PDV Extraction User Interface"

# color and other user preferences
COLOR_MAP_PREFERENCE = ""

class GUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)

        container.pack_propagate(0)

        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.spec = None
        self.plottedSpec = None

        self.frames = {}


        # if you want to add a new page/Tkinter window
        # append the page class name to the back of this list
        pages = [StartPage]


        for F in pages:
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    #METHODS


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.filename = None
        self.tkobjects = []

        label = tk.Label(self, text="Los Alamos National Laboratory, Velocity Extraction GUI", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        openButton = tk.Button(self, text="Select File", command = lambda: self.openFile())
        openButton.pack()

        print("application ready")


    def openFile(self):

        if self.filename is None:
            self.open_file_helper()
        else:
            for tkobject in self.tkobjects:
                tkobject.destroy()
            self.open_file_helper()
        

    def open_file_helper(self):

        self.filename =  filedialog.askopenfilename(initialdir=PATH_TO_DIR, title="Select dig file", filetypes=(("dig files","*.dig"),("all files","*.*")))
        digfile_name = os.path.basename(os.path.normpath(self.filename))

        try:
            sp = Spectrogram(digfile_name)
            assert isinstance(digfile_name, str)
            assert isinstance(sp, Spectrogram)
            display_text = "loaded dig file: "+digfile_name
            successlabel = tk.Label(self, text=display_text, font=LARGE_FONT)

            successlabel.pack(pady=10,padx=10)
            dispButton = tk.Button(self, text="Analyze Spectrogram", command = lambda: self.analyzeDig(sp))
            
            self.tkobjects.append(dispButton)
            self.tkobjects.append(successlabel)

            dispButton.pack()

        except TypeError:
            errorlabel = tk.Label(self, text="Error: file type is not dig format", font=LARGE_FONT)

            self.tkobjects.append(errorlabel)

            errorlabel.pack(pady=10,padx=10)
            return


    def analyzeDig(self, sp):

        # sgram = sp.spectrogram_no_log(0, 50e-6)
        sgram_log = sp.spectrogram(0,50e-6)

        intensities, interesting_velocites, baseline_velocity = sp.extract_velocities(sgram_log)

        # print(baseline_velocity)
        # print(type(intensities))
        # print(type(interesting_velocites))

        try: 
            assert isinstance(baseline_velocity, float)
            assert isinstance(interesting_velocites, list)
            assert isinstance(intensities, spectrogram.np.ndarray)
            baseline_text = "baseline velocity: " + str(baseline_velocity) + " m/s"
            baseline_label = tk.Label(self, text=baseline_text, font=LARGE_FONT)
            self.tkobjects.append(baseline_label)
            baseline_label.pack()

        except TypeError:
            error_label = tk.Label(self, text="Error: spectrogram data could not be extracted", font=LARGE_FONT)
            self.tkobjects.append(error_label)
            error_label.pack(pady=10,padx=10)
            return 

        dispButton = tk.Button(self, text="Display Spectrogram", command = lambda: self.displayDig(sp, sgram_log))
        self.tkobjects.append(dispButton)
        dispButton.pack()


    def displayDig(self, sp, sgram):

        axes = plt.axes()

        sp.tkplot(axes, sgram)

#         # plt.plot(velocities, intensities, "r")

#         # plt.xlim(1800, 2050)

#         # plt.show()



#         # f = Figure(figsize=(10,10), dpi=200)
#         # a = f.add_subplot(111)
#         # a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        

#         # canvas = FigureCanvasTkAgg(f, self)
#         # canvas.show()
#         # canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

#         # toolbar = NavigationToolbar2TkAgg(canvas, self)
#         # toolbar.update()
#         # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)



if __name__ == "__main__":

    app = GUI()
    app.geometry(WINDOW_SIZE)
    app.title(WINDOW_TITLE)

    app.mainloop()