import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    
    A major chord is three notes, played at the same time:
    (1) The root tone (f)
    (2) A major third, i.e., four semitones above f
    (3) A major fifth, i.e., seven semitones above f
    '''
    duration = 0.5  
    num_samples = int(duration * Fs)
    t = np.arange(num_samples)
    
    root = np.cos(2 * np.pi * f * t / Fs)
    
   
    major_third_freq = f * (2 ** (4 / 12))
    major_third = np.cos(2 * np.pi * major_third_freq * t / Fs)
    
    
    perfect_fifth_freq = f * (2 ** (7 / 12))
    perfect_fifth = np.cos(2 * np.pi * perfect_fifth_freq * t / Fs)
    
    x = root + major_third + perfect_fifth
    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    k = np.arange(N).reshape(-1, 1)  
    n = np.arange(N)                
    exponents = -1j * 2 * np.pi * k * n / N
    W = np.exp(exponents)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    N = len(x)
    X = np.fft.fft(x)
    magnitudes = np.abs(X[:N//2]) 
    freqs = np.fft.fftfreq(N, 1/Fs)[:N//2]
    

    indices = np.argsort(magnitudes)[-3:] 

    selected_freqs = freqs[indices]
    selected_freqs.sort()
    
    f1, f2, f3 = selected_freqs
    return f1, f2, f3
