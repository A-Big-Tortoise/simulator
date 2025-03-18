import numpy as np
import matplotlib.pyplot as plt

def sine_wave(max_amp, min_amp, samples, duty_cycle):
    f_hr = 0.5 / duty_cycle
    t = np.linspace(0, 1, samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * f_hr * t)[:int(samples * duty_cycle)]
    zeros = np.zeros(samples - int(samples * duty_cycle))
    return np.concatenate([sine_wave, zeros])


wave = sine_wave(512, 0, 410, 0.05)
plt.figure(figsize=(10, 3))
plt.plot(wave)
plt.savefig('/home/simulator/heartsim_easy/figs/sine_wave_base.png')
plt.close()


def square_wave(max_amp, min_amp, samples, duty_cycle):
    high_length = int(samples * duty_cycle)
    low_length = samples - high_length
    high_part = np.full(high_length, 1)
    low_part = np.zeros(low_length)
    return np.concatenate([high_part, low_part])


wave = square_wave(512, 0, 410, 0.05)
plt.figure(figsize=(10, 3))
plt.plot(wave)
plt.savefig('/home/simulator/heartsim_easy/figs/square_wave_base.png')
plt.close()