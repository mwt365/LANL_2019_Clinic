from tkinter import *
from tkinter import filedialog

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
        quitButton.place(x=0, y=0)
        openButton.place(x=0,y=0)



    #methods
    def client_exit(self):
        exit()

    def openFile(self):
        root.filename =  filedialog.askopenfilename(initialdir = "/",title = 
            "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

root = Tk()

#size of the window
root.geometry("400x300")

app = Window(root)
root.mainloop()  