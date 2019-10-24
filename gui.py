from tkinter import *
from tkinter import filedialog
from spectrogram import Spectrogram

class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)                 
        self.master = master
        self.init_window()

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget      
        self.master.title("GUI")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a button instance
        quitButton = Button(self, text="Quit",command=self.client_exit)
        openButton = Button(self, text="Select file", command=self.openFile)

        # placing the button on my window
        quitButton.pack(side=RIGHT)
        openButton.pack(side=LEFT)

    #variables
    spec = None


    #methods
    def client_exit(self):
        exit()

    def openFile(self):
        filename =  filedialog.askopenfilename(initialdir = "/documents",title = 
            "Select file",filetypes = (("dig files","*.dig"),("all files","*.*")))
        print(filename)
        self.spec = Spectrogram(filename)


root = Tk()

#size of the window
root.geometry("400x300")

app = Window(root)
root.mainloop()  