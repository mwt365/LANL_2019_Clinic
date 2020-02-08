#! /usr/bin/env python3

import xml_cm_files.cm_xml_to_matplotlib as cm
import os

DIR = os.path.join(os.path.split(__file__)[0], "xml_cm_files")

# make the Matplotlib compatible colormap

# Let's load all the colormaps
COLORMAPS = {}

try:
    for base, dirs, files in os.walk(DIR):
        for file in files:
            if file.endswith('.xml'):
                COLORMAPS[os.path.splitext(file)[0]] = cm.make_cmap(os.path.join(base[2:], file))
except FileNotFoundError:
    print("No folder named 'xml color map files' in the source directory.")

# mycmap = cm.make_cmap(os.path.join(DIR, '3w_gby.xml'))
# to use colormap: matplotlib.pyplot.imshow(your_image,
# cmap=matplotlib.pyplot.get_cmap(mycmap))

# cm.plot_cmap(mycmap)  # plot an 8 by 1 copy of the colormap
