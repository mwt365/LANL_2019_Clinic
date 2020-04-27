import matplotlib.pyplot as plt

def increaseReadability(title, ylab, xlab, titleSize, ylabSize, xlabSize, TickSize):

    plt.title(title, fontsize = titleSize)
    plt.ylabel(ylab, fontsize = ylabSize)
    plt.xlabel(xlab, fontsize = xlabSize)
    plt.xticks(fontsize = TickSize)
    plt.yticks(fontsize = TickSize)
    plt.show()
