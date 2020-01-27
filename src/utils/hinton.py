#!/usr/bin/env python
"""
Draws Hinton diagrams using matplotlib ( http://matplotlib.sf.net/ ).
Hinton diagrams are a handy way of visualizing weight matrices, using
colour to denote sign and area to denote magnitude.

By David Warde-Farley -- user AT cs dot toronto dot edu (user = dwf)
  with thanks to Geoffrey Hinton for providing the MATLAB code off of 
  which this is modeled.

Redistributable under the terms of the 3-clause BSD license 
(see http://www.opensource.org/licenses/bsd-license.php for details)

*I have modified this code slightly to adjust the visualization. Credit
is mainly to original authors.
"""

import numpy as np
import matplotlib.transforms
import matplotlib.colorbar
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

def _blob(x, y, area, colour, val, textcolor="black"):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour)
    if val < 0.05:
        plt.text(x - 0.25, y - 0.2, "⁎", fontsize=5, color=textcolor, fontweight="bold")

def hinton(W, scale, xlabels, ylabels, maxscale=None, filename="testing-hinton-diagram.png"):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    """if plt.isinteractive():
        plt.ioff()
    """
    plt.clf()
    height, width = W.shape
    if not maxscale:
        maxscale = 2**np.ceil(np.log(np.max(np.abs(W)))/np.log(2))
        
    plt.fill(np.array([0, width, width, 0]), 
             np.array([0, 0, height, height]),
             '#d3d3d3')
    
    #plt.axis('off')
    plt.axis('equal')
    cur_axes = plt.gca()
    cur_axes.spines['top'].set_visible(False)
    cur_axes.spines['right'].set_visible(False)
    cur_axes.spines['bottom'].set_visible(False)
    cur_axes.spines['left'].set_visible(False)
    #cur_axes.xaxis.set_label_position("bottom")

    plt.tick_params(axis="both", which="both", length=0)
    plt.subplots_adjust(left=0.12, bottom=0.11, right=0.98, top=0.97, wspace=0.09, hspace=0.21)

    #cmap = cm.viridis_r
    #cmap_neg = cm.bone
    cmap = cm.coolwarm
    norm = Normalize(vmin=np.min(W), vmax=np.max(W))
    norm_r = Normalize(vmin=np.min(W), vmax=np.max(W))
    xticks = set()
    yticks = set()
    for x in range(width):
        for y in range(height):
            _x = x+1
            _y = y+1
            w = W[y, x]
            s = scale[y, x]
            if s  < 0:
                color = matplotlib.colors.to_hex(cmap(norm_r(w)))
            else:
                color = matplotlib.colors.to_hex(cmap(norm(w)))
            xticks.add(_x - 0.5)
            yticks.add(height - _y + 0.5)
            if s > 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, s/maxscale),
                      "#FFA69E",
                      w,
                       textcolor="#381D2A")
            elif s <= 0:
                _blob(_x - 0.5,
                      height - _y + 0.5, 
                      min(1, -s/maxscale), 
                      "#7FD1B9",
                      w,
                      textcolor="#381D2A")
    plt.xticks(list(xticks), xlabels, fontsize=4, rotation='vertical')
    #plt.xlabel(xlabels, labelpad=-1000)
    dx = 0.; dy = 1.9
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, plt.gcf().dpi_scale_trans)
    for label in cur_axes.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    plt.yticks(list(yticks), ylabels, fontsize=6)

    plt.tight_layout()

    plt.savefig(filename + ".png", dpi=1000)
    plt.savefig(filename + ".eps", dpi=1000)

    if reenable:
        plt.ion()
    
def convert_to_hex(colortuple):
    return '#{:02x}{:02x}{:02x}'.format(*colortuple)

if __name__ == "__main__":
    hinton(np.random.randn(20, 20))
    plt.title('Example Hinton diagram - 20x20 random normal')
    plt.show()