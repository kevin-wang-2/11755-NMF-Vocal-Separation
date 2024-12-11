#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Unsupervised NMF to try separating into 5 sources (normally Vocal, Rythm, Bass, Strings, Other)
"""

import numpy as np
import sklearn.decomposition
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import math

# Load Audio
HOP_LENGTH = 352
WIN_LENGTH = 1411
n_fft = 2048
n_mels = 128
n_mfcc = 40

def load_audio(filename, max_len=5000):
    # If mfcc already loaded, directly fetch mfcc
    try:
        data = np.load(filename.replace(".wav", ".npz"))
        if max_len == -1:
            return data['M'], data['phase'], data['sr']
        else:
            return data['M'][:, :max_len], data['phase'][:, :max_len], data['sr']
    except:
        pass
    audio, sr = librosa.load(filename, sr=None)

    # mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mels=n_mels, n_mfcc=n_mfcc, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

    spec = librosa.stft(audio, n_fft=n_fft, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    M = np.abs(spec)
    phase = spec / (M + 1e-8)

    # Save spectrum
    np.savez(filename.replace(".wav", ".npz"), M=M, phase=phase, sr=sr)

    if max_len == -1:
        return M, phase, sr
    else:
        return M[:, :max_len], phase[:, :max_len], sr

def save_audio(file_path, M, phase, sr):
    # Convert back to audio
    spec = M * phase
    audio = librosa.istft(spec, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    sf.write(file_path, audio, sr)
    print("Audio saved to: ", file_path)

    return audio

"""
    NMF Train Function and definition of KL divergence used in NMF training.
"""
def KL_divergence(V, W, H):
    WH = np.dot(W, H)
    return np.sum(V * np.log(V / (WH + 1e-8)) - V + WH)

def NMF_train(V, n_components=5, max_iter=1000):
    model = sklearn.decomposition.NMF(n_components=n_components, init='nndsvda', beta_loss='kullback-leibler', solver='mu', max_iter=max_iter, verbose=True)
    W = model.fit_transform(V)
    H = model.components_
    return W, H

"""
    ICA Train Function
"""
def ICA_train(V, n_components=5, max_iter=1000):
    model = sklearn.decomposition.FastICA(n_components=n_components, max_iter=max_iter)
    S = model.fit_transform(V)
    A = model.mixing_
    return S, A

"""
    Main Function
"""

def decompose_NMF(music_file):
    # Load Audio
    M, phase, sr = load_audio(music_file, max_len=-1)
    V = M

    # NMF
    W, H = NMF_train(V, n_components=36, max_iter=1000)
    print("W shape: ", W.shape)
    print("H shape: ", H.shape)

    # Reconstruct the audio using each basis
    for i in range(W.shape[1]):
        M_hat = np.dot(W[:, i].reshape(-1, 1), H[i].reshape(1, -1))
        save_audio(f"{music_file}_output_{i}.wav", M_hat, phase, sr)

    return W, H

if __name__ == "__main__":
    import sys
    decompose_NMF(sys.argv[2])