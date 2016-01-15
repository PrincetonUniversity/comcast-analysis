from __future__ import division
import pickle as pkl
import pandas as pd
import numpy as np
import os, sys
from collections import defaultdict
import datetime
import matplotlib
from python_latexify import latexify, format_axes
params = {#'backend': 'ps',
            #'text.latex.preamble': ['\usepackage{gensymb}'],
            'axes.labelsize': 8, # fontsize for x and y labels (was 10)
            'axes.titlesize': 8,
            'font.size': 8, # was 10
            'legend.fontsize': 8, # was 10
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            #'text.usetex': True,
            'figure.figsize': [3.39,2.095],
            'font.family': 'serif'
          }
matplotlib.rcParams.update(params)

LATEXIFY = 0
if LATEXIFY:
    matplotlib.rcParams.update(latexify())

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel('DEBUG')

CONVERT_OCTETS = 8 / (15 * 60 * 1024)    #BYTES TO kbps

folder = "final_dw/"
#folder = "final_up/"
DATA = "../final_processed/" + folder
OUTPUT = "../final_output/" + folder
PLOTS = "../plots/"+folder
if not os.path.exists(PLOTS):
    os.makedirs(PLOTS)
if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

df1 = pd.read_pickle(DATA + "test.pkl")
df2 = pd.read_pickle(DATA + "control.pkl")

df1_fix = df1[df1.index > pd.datetime(2014, 9, 30, 23, 59, 59)]
df2_fix = df2[df2.index > pd.datetime(2014, 9, 30, 23, 59, 59)]
MB = 10**6
df = pd.DataFrame( {'treatment':df1_fix.mean(1)/MB, 'control':df2_fix.mean(1)/MB})
df['month'] = df.index.month

from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes

# function for setting the colors of the box plots pairs
def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], marker='x', color='blue')
    setp(bp['whiskers'][1], marker='x', color='blue')
    #setp(bp['fliers'][0], color='blue')
    #setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], marker='x', color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], marker='o', markersize=4, color='red')
    setp(bp['whiskers'][3], marker='o', markersize=4, color='red')
    #setp(bp['fliers'][2], color='red')
    #setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], marker='o', markersize=4, color='red')

# Some fake data to plot
Oct = [df[df.month==10]['treatment'],  df[df.month==10]['control']]
Nov = [df[df.month==11]['treatment'],  df[df.month==11]['control']]
Dec = [df[df.month==12]['treatment'],  df[df.month==12]['control']]

fig, ax = plt.subplots(1,1)
hold(True)

# first boxplot pair
bp = ax.boxplot(Oct, positions = [1, 2], widths = 0.6)
setBoxColors(bp)

# second boxplot pair
bp = ax.boxplot(Nov, positions = [4, 5], widths = 0.6)
setBoxColors(bp)

# thrid boxplot pair
bp = ax.boxplot(Dec, positions = [7, 8], widths = 0.6)
setBoxColors(bp)

# set axes limits and labels
ax.set_xlim(0,9)
#ax.set_xlabel("Month")
ax.set_ylabel("Traffic Demand per subs. (MB)")
#ylim(0,9)
ax.set_xticklabels(['Oct', 'Nov', 'Dec'])
ax.set_xticks([1.5, 4.5, 7.5])

# draw temporary red and blue lines and use them to create a legend
hB, = plt.plot([1,1],'b-x')
hR, = plt.plot([1,1],'r-o')
ax.legend((hB, hR),('treatment', 'control'))
hB.set_visible(False)
hR.set_visible(False)

ax.grid(1)

fig.tight_layout()
format_axes(ax)

savefig(OUTPUT + 'boxplot_monthly_avg_demand_downlink.png')
savefig(OUTPUT + 'boxplot_monthly_avg_demand_downlink.pdf')