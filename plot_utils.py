import matplotlib as mpl 
import seaborn as sns
import numpy as np
from array import array
import ROOT

def set_matplotlib_palette(paletteName):
    n_colors = 255
    cmap = mpl.colormaps[paletteName]
    colors = cmap(np.linspace(0, 1, n_colors))
    stopsList = list(np.linspace(0, 1, n_colors-1))+[1]
    stops = array('d',stopsList)
    red = array('d', colors[:, 0])
    green = array('d', colors[:, 1])
    blue = array('d', colors[:, 2])
    ROOT.TColor.CreateGradientColorTable(n_colors, stops, red, green, blue, 255)
    ROOT.gStyle.SetNumberContours(255)

def get_discrete_matplotlib_palette(paletteName, n_colors=10):
    try:
        cmap = mpl.colormaps[paletteName]
        colors = cmap.colors
    except:
        colors = sns.color_palette(paletteName, n_colors=n_colors)
        print(colors)
    ROOTColorIndices = []
    ROOTColors = []
    for color in colors:
        idx = ROOT.TColor.GetFreeColorIndex()
        ROOTColors.append(ROOT.TColor(idx, color[0], color[1], color[2],"color%i" % idx))
        ROOTColorIndices.append(idx)
        
    return ROOTColorIndices, ROOTColors