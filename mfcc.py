"""
s(n) - signal
s_i(n) - framed signal, where i = to 'which frame' and n = 0->num_samples
s_i(k) - i is still frame, no idea what k is. this is the DFT
p_i(k) - is the power spectrum of frame i

working through http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/

"""
import math
import numpy as np 
import matplotlib.pyplot as plt

frame_length = 0.025 # anything from 0.02 - 0.04, but 25ms is standard
sig_samplerate = 16000
num_samples = frame_length * sig_samplerate
frame_step = 0.010 * sig_samplerate # 10 ms step, creates some overlap

# first num_samples frame starts at 0, the next at frame_step, the next at 2*frame_step, etc

# if not divisble by 2, pad it with 0 until it is

#  extract oen set of MFCC coefficients from EACH frame

def convert_to_mel(frequency):    
    return 1125 * math.log(1 + frequency / 700)

def convert_from_mel(mel):
    return 700*(math.exp(mel/1125) - 1)

def get_filterbanks():

    lower = 200
    upper = sig_samplerate // 2
    mel_low = convert_to_mel(lower)
    mel_high = convert_to_mel(upper)
    num_filter_banks = 26 # anywhere from 26 - 40

    mels = np.linspace(mel_low, mel_high, num_filter_banks+2)

    freqs = list()
    for mel in mels:
        freqs.append(convert_from_mel(mel))

    print(freqs)

    nfft = 512
    bins = list()
    for f in freqs:
        bins.append(math.floor((nfft+1) * f/sig_samplerate)) 

    print(bins)
    filter_bank = np.zeros([num_filter_banks,nfft//2+1])

    for j in range(0,num_filter_banks):
        for i in range(int(bins[j]), int(bins[j+1])):
            filter_bank[j,i] = (i - bins[j]) / (bins[j+1]-bins[j])
            
        for i in range(int(bins[j+1]), int(bins[j+2])):
            filter_bank[j,i] = (bins[j+2] - i) / (bins[j+2] - bins[j+1])

    print(filter_bank)
    plt.plot(filter_bank)
    plt.show()
