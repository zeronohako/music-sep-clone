import numpy as np
import librosa

N_FFT = 1024
HOP = 512
WIN = "hann"

def wave_to_spec(wav_stereo, sr=44100):
    """wav_stereo: (T,2) float32 in [-1,1] -> complex STFT (2,F,Frames), mags, phases"""
    S = []
    for ch in range(2):
        s = librosa.stft(wav_stereo[:, ch], n_fft=N_FFT, hop_length=HOP, window=WIN, center=True)
        S.append(s)
    S = np.stack(S, axis=0)           # (2,F,Frames)
    mag = np.abs(S)
    phase = np.angle(S)
    return S, mag, phase

def spec_to_wave(S, sr=44100):
    """S: complex (2,F,Frames) -> wav_stereo (T,2)"""
    outs = []
    for ch in range(2):
        outs.append(librosa.istft(S[ch], hop_length=HOP, window=WIN, center=True, length=None))
    # pad to same length and stack
    T = max(len(x) for x in outs)
    outs = [np.pad(x, (0, T-len(x))) for x in outs]
    wav = np.stack(outs, axis=-1).astype(np.float32)  # (T,2)
    return np.clip(wav, -1.0, 1.0)
