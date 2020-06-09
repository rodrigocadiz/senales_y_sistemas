import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def rect(t):
    """Función rect"""
    return (np.abs(t)<0.5).astype(float)

def triang(t):
    """Función triángulo"""
    return (1-np.abs(t)) * (np.abs(t)<1).astype(float)

def periodic_triang(t):
    """Función triángulo periódico"""
    return triang(np.mod(t+1, 2)-1)
    
def plotRealSequence(seq,title,xlabel='n',ylabel='',continuous=False):
    samples = np.arange(len(seq))
    plt.stem(samples,seq,use_line_collection=True,linefmt='C0--', markerfmt='C0o', basefmt='C0-')
    if (continuous == True):
        plt.plot(samples,seq)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plotComplexSequence(seq,title,xlabel='n',ylabel='',continuous=False):
    samples = np.arange(len(seq))
    plt.stem(samples,np.real(seq),use_line_collection=True,linefmt='C0--', markerfmt='C0o', basefmt='C0-',label='Real')
    plt.stem(samples,np.imag(seq),use_line_collection=True,linefmt='C1-', markerfmt='C1s', basefmt='C1-',label='Imag')
    if (continuous == True):
        plt.plot(np.real(samples,seq))
        plt.plot(np.imag(samples,seq))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    
def plotSequence(seq,title,xlabel='n',ylabel='',continuous=False):
    if (seq.dtype == complex):
        plotComplexSequence(seq,title,xlabel,ylabel,continuous)
    else:
        plotRealSequence(seq,title,xlabel,ylabel,continuous)
    
def circularConvolution(f,h):
    """Evaluación de la convolución circular de f con h"""
    if (f.dtype == complex or h.dtype == complex):
        r = np.zeros(len(f),dtype=complex)
    else:
        r = np.zeros(len(f))     
    for k in range(len(f)):
        for p in range(len(f)):
            r[k] = r[k] + (f[p] * h[k-p])
    return r

def DFT(f):
    """Evaluación numérica de la DFT de f"""
    N = len(f)
    return (1/N)*np.fft.fft(f)

def IDFT(f):
    """Evaluación numérica de la DFT inversa de f"""
    N = len(f)
    return N*np.fft.ifft(f)

def CFT(f, u, a, b):
    """Evaluación numérica de la FT de f para las frecuencias u"""    
    result = np.zeros(len(u), dtype=complex)
    
    # Loop sobre todas las frecuencias y cálculo de la integral
    for i, uu in enumerate(u):
        result[i] = complex_int(lambda t: f(t)*np.exp(-2j*np.pi*uu*t), a, b)
    return result

def complex_int(f, a, b):
    """Devuelve la integral definida de la función compleja f, entre a y b, mediante la regla de Simpson"""
    t = np.linspace(a, b, 2501)  
    x = f(t)
    return integrate.simps(y=x, x=t)