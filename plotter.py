import cm_xml_to_matplotlib as cm

mycmap = cm.make_cmap('7-section-muted.xml') #make the Matplotlib compatible colormap
## to use colormap: matplotlib.pyplot.imshow(your_image, cmap=matplotlib.pyplot.get_cmap(mycmap))

cm.plot_cmap(mycmap) #plot an 8 by 1 copy of the colormap
