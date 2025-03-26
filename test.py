import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal
from scipy.signal import resample


def apply_amp(wave, max_amp, min_amp):
    return min_amp + ((wave - np.min(wave)) / (np.max(wave) - np.min(wave))) * (max_amp - min_amp)




# def sine_wave(max_amp, min_amp, samples, duty_cycle):
#     f_hr = 0.5 / duty_cycle
#     t = np.linspace(0, 1, samples, endpoint=False)
#     sine_wave = np.sin(2 * np.pi * f_hr * t)[:int(samples * duty_cycle)]
#     zeros = np.zeros(samples - int(samples * duty_cycle))
#     wave = np.concatenate([sine_wave, zeros])
#     return apply_amp(wave, max_amp, min_amp)


# wave = sine_wave(512, 0, 410, 0.75)
# plt.figure(figsize=(10, 3))
# plt.plot(wave)
# plt.savefig('/home/simulator/heartsim_easy/figs/sine_wave_base.png')
# plt.close()


# def square_wave(max_amp, min_amp, samples, duty_cycle):
#     high_length = int(samples * duty_cycle)
#     low_length = samples - high_length
#     high_part = np.full(high_length, 1)
#     low_part = np.zeros(low_length)
#     wave = np.concatenate([high_part, low_part])
#     return apply_amp(wave, max_amp, min_amp)


# wave = square_wave(512, 0, 410, 0.05)
# plt.figure(figsize=(10, 3))
# plt.plot(wave)
# plt.savefig('/home/simulator/heartsim_easy/figs/square_wave_base.png')
# plt.close()



def sine_wave_base(samples, duty_cycle, hrv=None):
    f_hr = 0.5 / duty_cycle
    t = np.linspace(0, 1, samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * f_hr * t)[:int(samples * duty_cycle)]
    zeros = np.zeros(samples - int(samples * duty_cycle))
    wave = np.concatenate([sine_wave, zeros])
    if hrv:
        hrv_len = int(samples * random.uniform(-1 * hrv, hrv))
        # print(hrv_len)
        if hrv_len > 0: 
            wave = np.concatenate([wave, np.zeros(hrv_len)])
        elif hrv_len < 0:
            wave = wave[:hrv_len]
    return wave



def sine_gen_with_rr_irr(min_amp, max_amp, samples, duty_circle, duration, hr, rr, rr_step, hrv):
    hr_num = int(duration * hr / 60)
    # rr_num = int(duration * rr / 60) + 1
    

    hr_waves = [sine_wave_base(samples, duty_circle, hrv) for _ in range(hr_num)]
    hr_waves_len = [len(wave) for wave in hr_waves]
    hr_waves = np.concatenate(hr_waves)
    print(f'len hr waves: {len(hr_waves)}')
    print(f'desried hr waves: {samples * duration}, actual hr waves: {len(hr_waves)}')

    rr_waves = generate_rr_wave(rr, samples, duration)
    print(f'len rr waves: {len(rr_waves)}')
    rr_waves_discrete = np.zeros_like(rr_waves)

    begin = 0
    end = hr_waves_len[0]
    for hr_wave_len in hr_waves_len:
        end = begin + hr_wave_len
        rr_waves_discrete[begin:end] = rr_waves[(2*begin + end) // 3]
        begin = end

    # if hr <140: val = val-1
    # scaling_factors = generate_increasing_amplitude_wave_array(val, rr_step)
    # rsa = np.tile(wave, len(scaling_factors)) * np.repeat(scaling_factors, len(wave))
    # wave_f = np.tile(rsa,reps)

    # new_length = duration*samples
    # hr_waves = np.interp(np.linspace(0, len(hr_waves), samples * duration), np.arange(len(hr_waves)), hr_waves)
    

    wave = apply_amp(hr_waves, max_amp, min_amp)
    return wave, rr_waves_discrete


def sine_gen_with_rr_irr_v2(min_amp, max_amp, samples, duty_circle, duration, hr, rr, rr_step):
    hr_num = int(duration * hr / 60)
    print(hr_num)
    
    hr_waves = [sine_wave_base(int(samples*60/hr), duty_circle) for _ in range(hr_num)]
    hr_waves_len = [len(wave) for wave in hr_waves]
    hr_waves = np.concatenate(hr_waves)
    hr_waves = resample(hr_waves, samples*duration)
    print(f'len hr waves: {len(hr_waves)}')
    print(f'desried hr waves: {samples * duration}, actual hr waves: {len(hr_waves)}')

    rr_waves = generate_rr_wave(rr, samples, duration)
    print(f'len rr waves: {len(rr_waves)}')
    rr_waves_discrete = np.zeros_like(rr_waves)

    begin = 0
    end = hr_waves_len[0]
    for hr_wave_len in hr_waves_len:
        end = begin + hr_wave_len
        rr_waves_discrete[begin:end] = rr_waves[(2*begin + end) // 3]
        begin = end

    hr_waves = hr_waves * rr_waves_discrete
    # hr_waves = np.interp(np.linspace(0, len(hr_waves), samples * duration), np.arange(len(hr_waves)), hr_waves)
    wave = apply_amp(hr_waves, max_amp, min_amp)
    return wave

# def generate_increasing_amplitude_wave_array(i,step_size):
#     step = step_size
#     wave_array = np.arange(1, 1 + step * i, step)
#     return wave_array


def generate_rr_wave(rr, samples, duration):
    t = np.linspace(0, duration, samples * duration, endpoint=False)
    rr_wave = signal.sawtooth(2 * np.pi * (rr/60) * t) / 2 + 1
    return rr_wave


duration = 60
rr = 7
hr = 40

wave = sine_gen_with_rr_irr_v2(min_amp=0, max_amp=512, samples=410, duty_circle=0.5, duration=duration, hr=hr, rr=rr, rr_step=0.1)


plt.figure(figsize=(10, 2))
plt.plot(wave)
plt.title(f'HR:{hr}, RR:{rr}')
plt.tight_layout()
plt.savefig(f'/home/simulator2/simulator/figs/sine_wave_irr_hr{hr}_rr{rr}_nohrv.png')
plt.close()
