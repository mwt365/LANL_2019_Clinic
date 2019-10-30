import tkinter as tk                # python 3
from tkinter import font  as tkfont # python 3
from tkinter import filedialog
from spectrogram import Spectrogram


from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np

class GUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.spec = None
        self.plottedSpec = None

        # self.frames = {}
        # for F in (StartPage, display_page, PageTwo): #Name the pages here
        #     page_name = F.__name__
        #     frame = F(parent=container, controller=self)
        #     self.frames[page_name] = frame

        #     # put all of the pages in the same location;
        #     # the one on the top of the stacking order
        #     # will be the one that is visible.
        #     frame.grid(row=0, column=0, sticky="nsew")

        # self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        window = tk.toplevel(page)

    #METHODS



class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is the start page", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        
        openButton = tk.Button(self, text="Load File", command = lambda: self.openFile())
        openButton.pack()

        dispButton = tk.Button(self, text="Display Spectrogram", command = lambda: self.displaySpec())
        dispButton.pack()

        # button1 = tk.Button(self, text="Go to Page One",
        #                     command=lambda: controller.show_frame("PageOne"))
        # button2 = tk.Button(self, text="Go to Page Two",
        #                     command=lambda: controller.show_frame("PageTwo"))
        # button1.pack()
        # button2.pack()

    def openFile(self):
        filename =  filedialog.askopenfilename(initialdir = "/documents",title = 
        # changing the title of our master widget      
            "Select file",filetypes = (("dig files","*.dig"),("all files","*.*")))
        print(filename)
        self.controller.spec = Spectrogram(filename)

    def displaySpec(self):
        self.controller.show_frame("display_page")




class display_page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is page 1", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()

        if self.controller.spec == None:
            print("no spectrogram")
            pass
        else:

            fig = Figure(figsize=(5, 4), dpi=100)
            t = np.arange(0, 3, .01)
            fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

            fig.add_subplot(self.controller.spec.plot())


            canvas = FigureCanvasTkAgg(fig, self)  # A tk.DrawingArea.
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            toolbar = NavigationToolbar2Tk(canvas, self)
            toolbar.update()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)



class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is page 2", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()


if __name__ == "__main__":
    app = GUI()
    app.mainloop()