import sys
import timeit
import numpy as np
from pydub import AudioSegment
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from utils import GENRE_LIST, GENRE_DIR

genre_list=GENRE_LIST

import scipy
from scipy.io.wavfile import read,write

def create_fftx_test(fn):
    """
        Creates the FFT features from the test files,
        saves them to disk, and returns the saved file name.
    """
    sample_rate, X1 = scipy.io.wavfile.read(fn)
    X2=np.reshape(X1,(-1,1))
    X=np.squeeze(X2)
    fft_features=abs(scipy.fft(X)[:2000])
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fftx"
    np.save(data_fn, fft_features)
    print ("Written ", data_fn)
    return data_fn


def read_fftx_test(test_file):
    """
        Reads the FFT features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    fft = np.load(test_file)
    num_fft = len(fft)
    X.append(fft[:2000])
    return np.array(X), np.array(y)


def convert_to_wav_format(file):

    if file.endswith("mp3"):
        song=AudioSegment.from_file(file,"mp3")
        song=song[:30000]
        song.export(file+"wav",format="wav")
    path=file+"wav"
    return path


def test_model_on_single_file(file_path):
    clf = joblib.load('C:\\Users\\hp\\Desktop\\project\\neurallbfgsdata.rar')
    X, y = read_fftx_test(create_fftx_test(convert_to_wav_file(file_path))+".npy")
    probs = clf.predict_proba(X)
    print ("\t".join(str(x) for x in genre_list))
    print ("\t".join(str("%.3f" % x) for x in probs[0]))
    probs=probs[0]
    max_prob = max(probs)
    for i,j in enumerate(probs):
        if probs[i] == max_prob:
            max_prob_index=i
    
    print (max_prob_index)
    predicted_genre = genre_list[max_prob_index]
    print ("\n\npredicted genre = ",predicted_genre)
    return predicted_genre


if __name__ == "__main__":

    print(genre_list)

    test = input("Enter the song's name ")
    test_file="C:\\Users\\hp\\Desktop\\project\\testfiles\\"+test+".mp3"
    predicted_genre = test_model_on_single_file(test_file)
