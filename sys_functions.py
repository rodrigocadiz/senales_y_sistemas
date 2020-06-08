import matplotlib.pyplot as plt
import numpy as np
    
def plotRealSequence(seq,title,xlabel='n',ylabel=''):
    samples = np.arange(len(seq))
    plt.stem(samples,seq,use_line_collection=True,linefmt='C0--', markerfmt='C0o', basefmt='C0-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plotComplexSequence(seq,title,xlabel='n',ylabel=''):
    samples = np.arange(len(seq))
    plt.stem(samples,np.real(seq),use_line_collection=True,linefmt='C0--', markerfmt='C0o', basefmt='C0-',label='Real')
    plt.stem(samples,np.imag(seq),use_line_collection=True,linefmt='C1-', markerfmt='C1s', basefmt='C1-',label='Imag')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    
def plotSequence(seq,title,xlabel='n',ylabel=''):
    if (seq.dtype == complex):
        plotComplexSequence(seq,title,xlabel,ylabel)
    else:
        plotRealSequence(seq,title,xlabel,ylabel)
    
def circularConvolution(f,h):
    if (f.dtype == complex or h.dtype == complex):
        r = np.zeros(len(f),dtype=complex)
    else:
        r = np.zeros(len(f))     
    for k in range(len(f)):
        for p in range(len(f)):
            r[k] = r[k] + (f[p] * h[k-p])
    return r

def DFT(f):
    N = len(f)
    return (1/N)*np.fft.fft(f)

def IDFT(f):
    N = len(f)
    return N*np.fft.ifft(f)