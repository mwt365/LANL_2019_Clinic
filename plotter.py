#! /usr/bin/env python3

import xml_cm_files.cm_xml_to_matplotlib as cm # Due to the updated folder structure.
import os

DIR = os.path.join(os.path.split(__file__)[0], "xml_cm_files")
# This should be pulling the folder structure from the current file

# make the Matplotlib compatible colormap

# Let's load all the colormaps
COLORMAPS = {}

try:
    for file in os.listdir(DIR):
        base, ext = os.path.splitext(file)
        if ext == '.xml':
            COLORMAPS[base] = cm.make_cmap(os.path.join(DIR, file))
except FileNotFoundError:
    print("No folder named 'xml color map files' in the source directory.")

# mycmap = cm.make_cmap(os.path.join(DIR, '3w_gby.xml'))
# to use colormap: matplotlib.pyplot.imshow(your_image,
# cmap=matplotlib.pyplot.get_cmap(mycmap))

# cm.plot_cmap(mycmap)  # plot an 8 by 1 copy of the colormap
