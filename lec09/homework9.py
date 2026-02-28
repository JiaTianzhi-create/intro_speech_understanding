import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)
    

    num_frames = (len(waveform) - frame_length) // step + 1
    frames = np.lib.stride_tricks.sliding_window_view(
        waveform, frame_length)[::step]
    

    energies = np.sum(frames ** 2, axis=1)
    max_energy = np.max(energies)
    thresh = 0.1 * max_energy
    

    voiced = energies > thresh
    

    segments = []
    start = None
    for i in range(len(voiced)):
        if voiced[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                seg_start = start * step
                seg_end = (i - 1) * step + frame_length
                segments.append(waveform[seg_start:seg_end])
                start = None
    if start is not None:
        seg_start = start * step
        seg_end = (len(voiced) - 1) * step + frame_length
        segments.append(waveform[seg_start:seg_end])
    
    return segments


def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    models = []
    frame_length = int(0.004 * Fs)   # 4ms
    step = int(0.002 * Fs)           # 2ms
    
    for seg in segments:
        if len(seg) < frame_length:
            continue 
        
        # Pre-emphasis
        preemph = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])
        
     
        num_frames = (len(preemph) - frame_length) // step + 1
        frames = np.lib.stride_tricks.sliding_window_view(
            preemph, frame_length)[::step]
        
     
        mstft = np.abs(np.fft.fft(frames, axis=1))
        
        low_half = mstft[:, :frame_length // 2]
        

        model = np.mean(low_half, axis=0)
        models.append(model)
    
    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)
    
    Y = len(models)   
    K = len(test_models)  
    sims = np.zeros((Y, K))
    
    for i in range(Y):
        for j in range(K):
            a = models[i]
            b = test_models[j]
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                sims[i, j] = 0
            else:
                sims[i, j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    

    test_outputs = [labels[np.argmax(sims[:, j])] for j in range(K)]
    
    return sims, test_outputs
