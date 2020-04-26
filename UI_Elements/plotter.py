#! /usr/bin/env python3

import os
import UI_Elements.xml_cm_files.cm_xml_to_matplotlib as cm # Due to the updated folder structure.


DIR = os.path.join(os.path.split(__file__)[0], "xml_cm_files")
# This should be pulling the folder structure from the current file

# make the Matplotlib compatible colormap

# Let's load all the colormaps
COLORMAPS = {}

try:
    for base, dirs, files in os.walk(DIR):
        for file in files:
            if file.endswith('.xml'):
                try:
                    path = os.path.join(base, file)
                    COLORMAPS[os.path.splitext(file)[0]] = cm.make_cmap(path)
                except Exception as eeps:
                    print(f"No luck with {path}")
except FileNotFoundError:
    print("No folder named 'xml color map files' in the source directory.")

DEFMAP = '3w_gby'
if '3w_gby' not in COLORMAPS:
	DEFMAP = list(COLORMAPS.keys())[0] # Just give me the first colormap in these colormaps.

