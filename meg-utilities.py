import math

"""

Stuff that helps me do signal/audio processing.

"""

def get_fft_hop(nfft=4096, overlap=0.75):
    hop = nfft-(nfft*overlap)

    return hop

# Given a patch size / data length, what's the ideal FFT window size?
def get_fft_window(data_len):
    window = data_len #TODO

    return window

# Say I want an fft with 75% overlap and a 4096 window, and my network requires the number of
# frames to == some value.
def get_patch_size(num_frames, nfft=4096, hop=1024):
    patch_size = (num_frames-1) * hop + nfft

    return patch_size

# Tells you the width of each bin in the FFT output (Hz)
def get_fft_resolution(sample_rate, nfft):
    resolution  = sample_rate // nfft

    return resolution 

def get_conv_output(input_size_w, input_size_h=None, specified_output_depth=None, filter_size=3, padding_size=0, stride=1):
    if input_size_h == None:
        input_size_h = input_size_w
    
    W_out = (input_size_w - filter_size + (2*padding_size) / (stride + 1)
    H_out = (input_size_h - filter_size + (2*padding_size)) / (stride + 1)

    print('Output size: {0} x {1} x {2}'.format(specified_output_depth, W_out, H_out))

def get_padding_size(filter_size):
    # stride == 1
    padding = (filter_size-1) // 2
    print('Padding: {0}'.format(padding))

def get_next_pow_2(x):
    power = math.ceil(math.log(x)/math.log(2))
    
    return 2**power

def get_prev_pow_2(x):
    power = math.floor(math.log(x)/math.log(2))

    return 2**power