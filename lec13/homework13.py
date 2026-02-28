import numpy as np

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    nframes = (len(speech) - frame_length) // frame_skip + 1
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]
        
        # 计算自相关系数
        r = np.correlate(frame, frame, mode='full')[frame_length-1:frame_length+order]
        
        # Levinson-Durbin 算法求 LPC 系数
        a = np.zeros(order + 1)
        a[0] = 1.0
        E = r[0]
        
        for k in range(1, order + 1):
            lambda_k = -np.dot(a[:k], r[1:k+1]) / E
            a_new = np.zeros(order + 1)
            a_new[0] = 1.0
            a_new[1:k+1] = a[1:k] + lambda_k * a[k-1::-1]
            a_new[k] = lambda_k
            a = a_new
            E = E * (1 - lambda_k**2)
        
        A[i] = a
        
        # 计算残差（excitation）
        residual = np.convolve(frame, a, mode='full')[:frame_length]
        excitation[i] = residual
    
    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    nframes = A.shape[0]
    frame_length = e.shape[1]
    total_length = (nframes - 1) * frame_skip + frame_length
    synthesis = np.zeros(total_length)
    
    for i in range(nframes):
        start = i * frame_skip
        a = A[i]
        residual = e[i]
        
        # LPC 合成滤波器：1 / A(z)
        for n in range(frame_length):
            synth_val = residual[n]
            for k in range(1, len(a)):
                if n - k >= 0:
                    synth_val -= a[k] * synthesis[start + n - k]
            synthesis[start + n] = synth_val
    
    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    nframes = excitation.shape[0]
    frame_length = excitation.shape[1]
    
    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)
    
    for i in range(nframes):
        frame = excitation[i]
        gain[i] = np.std(frame)  # 或 np.sqrt(np.mean(frame**2))
        if gain[i] == 0:
            gain[i] = 1e-6  # 避免除零
        
        # 机器人声：固定周期 T0 的脉冲激励
        robot_frame = np.zeros(frame_skip)
        robot_frame[::T0] = 1.0 / gain[i]  # 归一化增益
        
        e_robot[i * frame_skip:(i+1) * frame_skip] = robot_frame
    
    return gain, e_robot
