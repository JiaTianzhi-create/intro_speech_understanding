import numpy as np
import torch, torch.nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    
    '''
    # Pre-emphasis
    preemph = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    
    # Spectrogram params
    frame_length_spec = int(0.004 * Fs)  # 4ms
    step_spec = int(0.002 * Fs)  # 2ms
    
    # 分帧 for spectrogram
    num_frames = (len(preemph) - frame_length_spec) // step_spec + 1
    frames_spec = np.lib.stride_tricks.sliding_window_view(preemph, frame_length_spec)[::step_spec]
    
    # Magnitude STFT
    mstft = np.abs(np.fft.fft(frames_spec, axis=1))
    
    # Low-frequency half
    features = mstft[:, :frame_length_spec // 2]
    
    # VAD params
    frame_length_vad = int(0.025 * Fs)  # 25ms
    step_vad = int(0.01 * Fs)  # 10ms
    
    # 分帧 for VAD
    num_frames_vad = (len(waveform) - frame_length_vad) // step_vad + 1
    frames_vad = np.lib.stride_tricks.sliding_window_view(waveform, frame_length_vad)[::step_vad]
    
    # Energies for VAD
    energies = np.sum(frames_vad ** 2, axis=1)
    max_energy = np.max(energies)
    thresh = 0.1 * max_energy
    
    # Voiced frames
    voiced = energies > thresh
    
    # Assign labels to segments, repeat each label 5 times per frame? Wait, adjust to match num_frames
    labels = np.zeros(num_frames, dtype=int)
    label = 0
    start = None
    for i in range(len(voiced)):
        if voiced[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                # Map VAD frames to spectrogram frames
                vad_start_sample = start * step_vad
                vad_end_sample = i * step_vad + frame_length_vad
                spec_start_frame = max(0, vad_start_sample // step_spec)
                spec_end_frame = min(num_frames, vad_end_sample // step_spec)
                labels[spec_start_frame:spec_end_frame] = label
                label += 1
                start = None
    if start is not None:
        vad_start_sample = start * step_vad
        vad_end_sample = len(waveform)
        spec_start_frame = max(0, vad_start_sample // step_spec)
        spec_end_frame = num_frames
        labels[spec_start_frame:spec_end_frame] = label
    

    
    return features, labels

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step.
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    NFEATS = features.shape[1]
    NLABELS = 1 + int(np.max(labels))
    
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(NFEATS),
        torch.nn.Linear(NFEATS, NLABELS)
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    
    features_t = torch.from_numpy(features).float()
    labels_t = torch.from_numpy(labels).long()
    
    lossvalues = np.zeros(iterations)
    
    for i in range(iterations):
        optimizer.zero_grad()
        outputs = model(features_t)
        loss = loss_fn(outputs, labels_t)
        loss.backward()
        optimizer.step()
        lossvalues[i] = loss.item()
    
    return model, lossvalues

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    features_t = torch.from_numpy(features).float()
    with torch.no_grad():
        outputs = model(features_t)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).detach().numpy()
    return probabilities