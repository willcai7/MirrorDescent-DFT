import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(silent=False):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if silent:
        return tempTimeInterval
    else:
        print( "Elapsed time: %f seconds." %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(True)
