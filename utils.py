
import os
import sys

from matplotlib import pylab
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")

for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)

# Put your directory to the different music genres here
GENRE_DIR = "C:\\Users\\hp\\Desktop\\project\\genre_dataset"
GENRE_LIST = ["bollypop", "carnatic", "ghazal", "semiclassical",  "sufi"]

# Put your directory to the test dir here
#TEST_DIR hasn't been set as of now. 

#The confusion matrix function will also be written here. pylab will be used for it.