"""
##################
#### Pulse.py ####
##################

Based on read_video_and_extract_roi.py
Gives estimate of pulse from either video file or text file

Requirements:
* Probably ffmpeg. Install it through your package manager.
* numpy
* OpenCV python package.
    For some reason, the default packages in Debian/Raspian (python-opencv and
    python3-opencv) seem to miss some important features (selectROI not present
    in python3 package, python2.7 package not compiled with FFMPEG support), so
    install them (locally for your user) using pip:
    - pip install opencv-python
    - (or pip3 install opencv-python)
* pyplot
    Required when using the -plot option
* scipy
"""

import sys
import numpy as np
import scipy.signal as sp
from sys import exit

# Constants
CROP_SIZE = 128

BPM_MIN = 30
BPM_MAX = 230

# Variables
source_file = ""

# Utility functions
def autocorrelation(x):
    result = np.correlate(x, x, mode="full")
    return result[int(len(result)/2):]

def spectrum(x):
    result = np.abs(np.fft.fft(x))
    return result[:int(len(result) / 2)]

def spectrumIndexToBPM(index, length, fps):
    return index / length * fps / 2 * 60

def hpfilter(signal):
    filtered_signal = np.zeros(len(signal) - 1)
    for i in range(0, len(signal) - 1):
         filtered_signal[i] = signal[i + 1] - signal[i];
    return filtered_signal


# CLI options
if len(sys.argv) > 1:
    source_file = sys.argv[1]
else:
    print("[ERROR] " + sys.argv[0] + " needs a target file.\nTry writing '" + sys.argv[0] + " [filename]'.")
    exit()

# Define color channels
signal_red = []
signal_green = []
signal_blue = []

sample_num = 0
sample_rate = 40 # Default value

#################################
##### Read from source file #####
#################################

if (source_file.endswith(".txt")):
    # Read data from text file
    print("Reading data from txt file.")

    mean_signal = np.loadtxt(source_file)
    sample_num = len(mean_signal)

    signal_red = np.zeros(sample_num)
    signal_green = np.zeros(sample_num)
    signal_blue = np.zeros(sample_num)    

    for i in range(0, len(mean_signal)):
        signal_red[i] = mean_signal[i][0]
        signal_green[i] = mean_signal[i][1]
        signal_blue[i] = mean_signal[i][2]
    
elif (source_file.endswith(".mp4")):
    # Read video file
    
    # Needs cv2 module for this
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV is not installed. Please install it.")
        exit()

    cap = cv2.VideoCapture(source_file, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("[ERROR] Could not open input file. Wrong filename, or your OpenCV package might not be built with FFMPEG support. See docstring of this Python script.")
        exit()

    sample_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = cap.get(cv2.CAP_PROP_FPS)

    mean_signal = np.zeros((sample_num, 3))

    std_pixels = np.zeros(sample_num)

    # Loop through the video
    count = 0
    while cap.isOpened():
        ret, frame = cap.read() #'frame' is a normal numpy array of dimensions [height, width, 3], in order BGR
        if not ret:
            break

        if count == 0:
            frame_width = frame.shape[0]
            frame_height = frame.shape[1]
            region_of_interest = [int((frame_width - CROP_SIZE) / 2), int((frame_height - CROP_SIZE) / 2), CROP_SIZE, CROP_SIZE]
        #    window_text = 'Select ROI by dragging the mouse, and press SPACE or ENTER once satisfied.'
        #    ROI = cv2.selectROI(window_text, frame) #ROI contains: [x, y, w, h] for selected rectangle
        #    cv2.destroyWindow(window_text)
            print("Looping through video.")

        cropped_frame = frame[region_of_interest[1]:region_of_interest[1] + region_of_interest[3], region_of_interest[0]:region_of_interest[0] + region_of_interest[2], :]
        mean_signal[count, :] = np.mean(cropped_frame, axis=(0,1))
        
        mean_pixels = np.mean(cropped_frame, axis=2)
        std_pixels[count] = np.mean(mean_signal[count, :]) / np.std(mean_pixels)
        
        count = count + 1
        print("\rFrame " + str(count) + "/" + str(sample_num), end="")
        
    print()
    cap.release()

    signal_red = mean_signal[:, 2]
    signal_green = mean_signal[:, 1]
    signal_blue = mean_signal[:, 0]

    print("Image SNR: " + str(np.mean(std_pixels)))

    if "-save" in sys.argv:
        output_file = source_file[:-4] + ".txt"
        np.savetxt(output_file, np.flip(mean_signal, 1))
        print("Data saved to '" + output_file + "', fps = " + str(sample_rate) + " frames/second")

else:
    print("[ERROR] File type not supported")
    exit()

########################
##### Analyze data #####
########################

# Design band pass filter
nyq = sample_rate / 2
low = 30 / 60 / nyq
high = 230 / 60 / nyq
b, a = sp.butter(5, [low, high], btype='band')

# Removes DC-level and filters the signal with previously designed filter.
filtered_signal_red = sp.lfilter(b, a, signal_red - np.mean(signal_red))
filtered_signal_green = sp.lfilter(b, a, signal_green - np.mean(signal_green))
filtered_signal_blue = sp.lfilter(b, a, signal_blue - np.mean(signal_blue))

# Puls
auto_red = autocorrelation(filtered_signal_red)
auto_green = autocorrelation(filtered_signal_green)
auto_blue = autocorrelation(filtered_signal_blue)

spectrum_red = spectrum(auto_red)
spectrum_green = spectrum(auto_green)
spectrum_blue = spectrum(auto_blue)
spectrum_length = len(spectrum_red)

bpm_red = spectrumIndexToBPM(np.argmax(spectrum_red), spectrum_length, sample_rate)
bpm_green = spectrumIndexToBPM(np.argmax(spectrum_green), spectrum_length, sample_rate)
bpm_blue = spectrumIndexToBPM(np.argmax(spectrum_blue), spectrum_length, sample_rate)

print("-- Pulse --")
print("Red: " + str(bpm_red) + " BPM")
print("Green: " + str(bpm_green) + " BPM")
print("Blue: " + str(bpm_blue) + " BPM")

# Signal-Noise-Ratio
peak_height_red = np.max(spectrum_red)
peak_height_green = np.max(spectrum_green)
peak_height_blue = np.max(spectrum_blue)

average_energy_red = np.mean(spectrum_red)
average_energy_green = np.mean(spectrum_green)
average_energy_blue = np.mean(spectrum_blue)


std_signal_red = np.std(filtered_signal_red)
std_signal_green = np.std(filtered_signal_green)
std_signal_blue = np.std(filtered_signal_blue)

std_noise_red = np.std(hpfilter(filtered_signal_red))
std_noise_green = np.std(hpfilter(filtered_signal_green))
std_noise_blue = np.std(hpfilter(filtered_signal_blue))

snr_red = snr_green = snr_blue = old_snr_red = old_snr_green = old_snr_blue = 0

if not std_noise_red == 0:
    old_snr_red = std_signal_red / std_noise_red
if not std_noise_green == 0:
    old_snr_green = std_signal_green / std_noise_green
if not std_noise_blue == 0:
    old_snr_blue = std_signal_blue / std_noise_blue

if not average_energy_red == 0:
    snr_red = peak_height_red / average_energy_red
if not average_energy_green == 0:
    snr_green = peak_height_green / average_energy_green
if not average_energy_blue == 0:
    snr_blue = peak_height_blue / average_energy_blue

print("-- Pulse SNR --")
print("Red: " + str(snr_red) + " (old: " + str(old_snr_red) + ")")
print("Green: " + str(snr_green) + " (old: " + str(old_snr_green) + ")")
print("Blue: " + str(snr_blue) + " (old: " + str(old_snr_blue) + ")")

print("-- Conclusion --")
best_color = ""
best_bpm = ""
if (snr_red > snr_green and snr_red > snr_blue):
    best_color = "red"
    best_bpm = bpm_red
elif (snr_green > snr_red and snr_green > snr_blue):
    best_color = "green"
    best_bpm = bpm_green
else:
    best_color = "blue"
    best_bpm = bpm_blue
print("Based on signal-to-noise ratio, " + best_color + " seems to be the best estimate, with " + str(best_bpm) + " BPM.")

if "-plot" in sys.argv:
    try:
        import matplotlib.pyplot as pp
    except ImportError:
        print("You need matplotlib to plot.")
        exit()

    pp.subplot(3, 4, 1)
    pp.plot(signal_red, "r")
    pp.subplot(3, 4, 2)
    pp.plot(filtered_signal_red, "r")
    pp.subplot(3, 4, 3)
    pp.plot(auto_red, "r")
    pp.subplot(3, 4, 4)
    pp.plot(spectrum_red, "r")
    
    pp.subplot(3, 4, 5)
    pp.plot(signal_green, "g")
    pp.subplot(3, 4, 6)
    pp.plot(filtered_signal_green, "g")
    pp.subplot(3, 4, 7)
    pp.plot(auto_green, "g")
    pp.subplot(3, 4, 8)
    pp.plot(spectrum_green, "g")
    
    pp.subplot(3, 4, 9)
    pp.plot(signal_blue, "b")
    pp.subplot(3, 4, 10)
    pp.plot(filtered_signal_blue, "b")
    pp.subplot(3, 4, 11)
    pp.plot(auto_blue, "b")
    pp.subplot(3, 4, 12)
    pp.plot(spectrum_blue, "b")
    
    output_plot = source_file[:-4] + ".png"
    pp.savefig(output_plot)
    pp.show()
    

print("Done")