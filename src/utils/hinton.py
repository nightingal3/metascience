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
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def _blob(x, y, area, colour):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
    plt.fill(xcorners, ycorners, colour)

def hinton(W, scale, xlabels, ylabels, maxscale=None):
    """
    Draws a Hinton diagram for visualizing a weight matrix. 
    Temporarily disables matplotlib interactive mode if it is on, 
    otherwise this takes forever.
    """
    reenable = False
    if plt.isinteractive():
        plt.ioff()
    
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

    plt.tick_params(axis="both", which="both", length=0)

    minscale = np.min(scale)
    cmap = cm.viridis
    norm = Normalize(vmin=np.min(W), vmax=np.max(W))
    xticks = set()
    yticks = set()
    for x in range(width):
        for y in range(height):
            _x = x+1
            _y = y+1
            w = W[y, x]
            s = scale[y, x]
            print(minscale/s)
            xticks.add(_x - 0.5)
            yticks.add(height - _y + 0.5)
            if w > 0:
                _blob(_x - 0.5,
                      height - _y + 0.5,
                      min(1, maxscale/s),
                      matplotlib.colors.to_hex(cmap(norm(w))))
            elif w < 0:
                _blob(_x - 0.5,
                      height - _y + 0.5, 
                      min(1, maxscale/s), 
                      matplotlib.colors.to_hex(cmap(norm(w))))
    print(xticks)
    plt.xticks(list(xticks), xlabels)
    plt.yticks(list(yticks), ylabels)
    plt.savefig("testing-hinton-diagram.png")
    if reenable:
        plt.ion()
    
def convert_to_hex(colortuple):
    return '#{:02x}{:02x}{:02x}'.format(*colortuple)

if __name__ == "__main__":
    hinton(np.random.randn(20, 20))
    plt.title('Example Hinton diagram - 20x20 random normal')
    plt.show()