import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

time_step = 0.05
time_vec = np.arange(0, 2, time_step)
sig = np.sin(2 * np.pi * 5 * time_vec) + np.random.randn(time_vec.size)  # T = 1

sig_fft = fftpack.fft(sig)

Amplitude = np.abs(sig_fft)
power = Amplitude ** 2
Angle = np.angle(sig_fft)

sample_freq = fftpack.fftfreq(sig.size, d=time_step)

Amp_freq = np.array([Amplitude, sample_freq])

Amp_pos = Amp_freq[0, :].argmax()
peak_freq = Amp_freq[1, Amp_pos]

print(peak_freq)

plt.figure()
plt.plot(sample_freq, Amplitude)
plt.show()
