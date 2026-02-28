import numpy as np

def voiced_excitation(duration, F0, Fs):
    '''
    Create voiced speech excitation.
    '''
    T0 = int(np.round(Fs / F0))
    excitation = np.zeros(duration)
    excitation[::T0] = -1.0  # 用 -1.0 确保 float
    return excitation


def resonator(x, F, BW, Fs):
    '''
    Generate the output of a resonator.
    '''
    # 计算系数（标准二阶数字共振器）
    alpha = np.exp(-np.pi * BW / Fs)
    theta = 2 * np.pi * F / Fs
    
    a1 = 2 * alpha * np.cos(theta)
    a2 = -alpha ** 2
    
    b0 = 1.0
    
    y = np.zeros_like(x, dtype=float)  # 关键：dtype=float，确保浮点数组
    
    # 前两个样本特殊处理（避免索引越界）
    if len(x) > 0:
        y[0] = b0 * x[0]
    if len(x) > 1:
        y[1] = b0 * x[1] + a1 * y[0]
    
    # 递归从 n=2 开始
    for n in range(2, len(x)):
        y[n] = b0 * x[n] + a1 * y[n-1] + a2 * y[n-2]
    
    return y


def synthesize_vowel(duration, F0, F1, F2, F3, F4, BW1, BW2, BW3, BW4, Fs):
    '''
    Synthesize a vowel.
    '''
    excitation = voiced_excitation(duration, F0, Fs)
    y1 = resonator(excitation, F1, BW1, Fs)
    y2 = resonator(y1, F2, BW2, Fs)
    y3 = resonator(y2, F3, BW3, Fs)
    speech = resonator(y3, F4, BW4, Fs)
    return speech