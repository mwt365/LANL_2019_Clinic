#! /usr/bin/env python3

import cm_xml_to_matplotlib as cm
import os

DIR = "xml color map files"

# make the Matplotlib compatible colormap

# Let's load all the colormaps
COLORMAPS = {}

for file in os.listdir(DIR):
    base, ext = os.path.splitext(file)
    if ext == '.xml':
        COLORMAPS[base] = cm.make_cmap(os.path.join(DIR, file))


# mycmap = cm.make_cmap(os.path.join(DIR, '3w_gby.xml'))
# to use colormap: matplotlib.pyplot.imshow(your_image, cmap=matplotlib.pyplot.get_cmap(mycmap))

# cm.plot_cmap(mycmap)  # plot an 8 by 1 copy of the colormap
