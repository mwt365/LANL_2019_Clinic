#! /usr/bin/env python3

# Due to the updated folder structure.
import UI_Elements.xml_cm_files.cm_xml_to_matplotlib as cm
import os

parent = os.path.split(__file__)[0]
DIR = os.path.join(parent, "xml_cm_files")

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
    # Just give me the first colormap in these colormaps.
    DEFMAP = list(COLORMAPS.keys())[0]

