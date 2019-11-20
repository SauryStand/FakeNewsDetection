# -*- coding-utf-8 -*-
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def bar_plot(x, y, title, color):
    # Set up barplot
    plt.figure(figsize=(9, 5))
    g = sns.barplot(x, y, color=color)
    ax = g

    # Label the graph
    plt.title(title, fontsize=15)
    plt.xticks(fontsize=10)

    # Enable bar values
    # Code modified from http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for p in ax.patches:
        totals.append(p.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for p in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(p.get_width() + .3, p.get_y() + .38, \
                int(p.get_width()), fontsize=10)