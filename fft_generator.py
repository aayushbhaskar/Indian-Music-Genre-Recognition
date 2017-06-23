#This file contains the fft feature extractor

import sys
import os
import glob

import scipy
import numpy
from scipy.io.wavfile import read,write

from utils import GENRE_DIR, CHART_DIR, GENRE_LIST

def write_fftx(fft_features, fn):
    # the features will be written on separate files.
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fftx"

    numpy.save(data_fn, fft_features)
    print("Written ")

def create_fftx(fn):
    sample_rate, X1 = scipy.io.wavfile.read(fn)
    X2=numpy.reshape(X1,(-1,1))
    X=numpy.squeeze(X2)
    print(X.shape)
    fft_features = abs(scipy.fft(X)[:2000])
    write_fftx(fft_features, fn)

def read_fftx(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fftx.npy")
        file_list = glob.glob(genre_dir)
        assert(file_list), genre_dir
        for fn in file_list:
            fft_features = numpy.load(fn)

            X.append(fft_features[:2000])
            y.append(label)

    return numpy.array(X), numpy.array(y)

def plot_confusion_matrix(cm, genre_list, name, title):
    """
        Plots confusion matrices.
    """
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class', fontsize = 20)
    pylab.ylabel('True class', fontsize = 20)
    pylab.grid(False)
    pylab.show()
    pylab.savefig(os.path.join(CHART_DIR, "confusion_matrix_%s.png" % name), bbox_inches="tight")

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection( set(GENRE_LIST) ))
        break
    print ("Working with these genres --> ", traverse)
    print ("Starting fft generation")     
    for subdir, dirs, files in os.walk(GENRE_DIR):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("wav"):
                print(os.path.join(subdir,file))
                create_fftx(path)
                    
    stop = timeit.default_timer()
    print ("Total fft generation and feature writing time (s) = ", (stop - start)) 
