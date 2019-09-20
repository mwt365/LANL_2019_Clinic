import cm_xml_to_matplotlib as cm

mycmap = cm.make_cmap('./xml_cm_files/.xml') #make the Matplotlib compatible colormap
## to use colormap: matplotlib.pyplot.imshow(your_image, cmap=matplotlib.pyplot.get_cmap(mycmap))

cm.plot_cmap(mycmap) #plot an 8 by 1 copy of the colormap

cm.cmap_matrix(mycmap)