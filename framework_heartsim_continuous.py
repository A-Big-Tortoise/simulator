import numpy as np
import time
from smbus2 import SMBus
import math
import argparse
import busio
import board
import adafruit_mcp4725 as a
import struct
import time
import paho.mqtt.client as mqtt
import threading
from scipy import signal
import netifaces



"""
************************ Used for Label Upload ************************
"""


# def get_mac(interface='eth0'):
#     return netifaces.ifaddresses(interface)[netifaces.AF_LINK][0]['addr']


def pack_beddot_data(mac_addr, timestamp, data_interval, data):
    # First, convert the MAC address into byte sequence
    mac_bytes = bytes.fromhex(mac_addr.replace(':', ''))
    # Then, pack each data item sequentially
    packed_data = struct.pack("!BBBBBB", *mac_bytes)
    packed_data += struct.pack("H", len(data))  # data length
    packed_data += struct.pack("L", timestamp)  # timestamp
    packed_data += struct.pack("I", data_interval)  # data interval 

   # pack measurement data(Blood Oxygen)
    for item in data:
        packed_data += struct.pack("i", item)
    return packed_data


# def write_mqtt(hrdata, rrdata, timestamp, fs):
#     mac_addr=get_mac()
#     timestamp = int(timestamp * 1000000)
#     hrdata = np.int32(hrdata)
#     rrdata = np.int32(rrdata)
#     Ts = int((1/fs) * 1000000)
#     packed_data_1 = pack_beddot_data(mac_addr, timestamp, Ts, hrdata) # 10000 equates to fs=1/0.01=100
#     packed_data_2 = pack_beddot_data(mac_addr, timestamp, Ts, rrdata)

#     client = mqtt.Client()
#     # client.connect("yuantzg.com", 9183)
#     client.connect("sensorweb.us", 1883)
#     mqtt_thread = threading.Thread(target=lambda: client.loop_forever())
#     mqtt_thread.start()

#     mac_addr_str = mac_addr.replace(":", "")
#     client.publish(f"/UGA/{mac_addr_str}/hrlabel", packed_data_1, qos=1)
#     client.publish(f"/UGA/{mac_addr_str}/rrlabel", packed_data_2, qos=1)
#     print('Published')
#     return None
def write_mqtt(hrdata, rrdata, timestamp, fs):
    mac_addr="12:02:12:02:12:02"
    timestamp = int(timestamp * 1000000)
    hrdata = np.int32(hrdata)
    rrdata = np.int32(rrdata)
    Ts = int((1/fs) * 1000000)
    packed_data_1 = pack_beddot_data(mac_addr, timestamp, Ts, hrdata) # 10000 equates to fs=1/0.01=100
    packed_data_2 = pack_beddot_data(mac_addr, timestamp, Ts, rrdata)

    client = mqtt.Client()
    client.connect("sensorweb.us", 1883)
    mqtt_thread = threading.Thread(target=lambda: client.loop_forever())
    mqtt_thread.start()


    client.publish("/UGA/120212021202/hrlabel", packed_data_1, qos=1)
    client.publish("/UGA/120212021202/rrlabel", packed_data_2, qos=1)
    print('Published')
    return None

"""
************************ Used for Waveform Generation ************************
"""


def generate_increasing_amplitude_wave_array(i,step_size):
    # Create an array starting from 1.1, increasing by step up to length i
    step = step_size
    wave_array = np.arange(1, 1 + step * i, step)
    return wave_array


def pulse_base(min_val, max_val, samples, duration):
    fs = samples  # Sampling frequency in Hz
    t = np.linspace(0, duration, fs, endpoint=False)  # 1 second duration
    duty_cycle = 0.05  # 50% duty cycle
    f_hr = 1
    amp =1
    # Generate pulse waveform
    pulse_wave = (np.sin(2 * np.pi * f_hr * t) >= np.cos(np.pi * duty_cycle)).astype(int)
    pulse_wave= ((pulse_wave-np.min(pulse_wave))/(np.max(pulse_wave)-np.min(pulse_wave)) * amp)
    
    pulse_wave = signal.resample(pulse_wave,duration*samples)
    pulse_wave= min_val + ((pulse_wave - np.min(pulse_wave)) / (np.max(pulse_wave) - np.min(pulse_wave))) * (max_val - min_val)
    pulse_wave = abs(pulse_wave)
    return pulse_wave


def pulse_gen_with_rr(min_val, max_val, samples, duration, hr, rr, rr_step):
    wave_a = pulse_base(min_val, max_val, samples, 1)

    val = int(np.round(hr/rr))
    reps = int(np.round(rr/60*duration))

    ## For RR effect
    if hr < 120: val = val-1
    scaling_factors = generate_increasing_amplitude_wave_array(val, rr_step)
    rsa = np.tile(wave_a, len(scaling_factors)) * np.repeat(scaling_factors, len(wave_a))
    wave_f = np.tile(rsa,reps)

    # Without RR effect
    # reps = int(hr)
    # wave_f = np.tile(wave_a, reps)

    new_length = duration*samples
    wave = np.interp(np.linspace(0, len(wave_f)-1, new_length), np.arange(len(wave_f)), wave_f)
    wave = min_val + ((wave - np.min(wave)) / (np.max(wave) - np.min(wave))) * (max_val - min_val)
    wave = abs(wave)
    return wave



def sine_wave_base(samples, duty_cycle):
    f_hr = 0.5 / duty_cycle
    t = np.linspace(0, 1, samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * f_hr * t)[:int(samples * duty_cycle)]
    zeros = np.zeros(samples - int(samples * duty_cycle))
    wave = np.concatenate([sine_wave, zeros])
    return wave



def sine_gen_with_rr_dc(amp, samples, duty_cycle):

    f_hr = 1
    duration = 1
    sampling_rate = samples
    phase = 0

    val = int(duty_cycle*samples)
    rem = samples - val
    zer_array = np.zeros(rem)
  
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * f_hr * t - phase)
    wave = sine_wave

    # wave = signal.resample(wave, val)
    # wave = np.concatenate((zer_array, wave),axis=0)
    wave = np.where(wave < 0, 0, wave)

    return wave


def sine_gen_with_rr_v4(min_amp, max_amp, samples, duration, hr, rr, rr_step):
    min_val = min_amp
    max_val = max_amp

    duty_cycle = 0.75

    ## Select base sine wave - with or without Duty Cycle
    wave = sine_gen_with_rr_dc(max_val, samples, duty_cycle)
    # wave = sine_wave_base(samples, 0.5)

    ### For RR effect uncomment this
    val = int(np.round(hr/rr))
    reps = int(np.round(rr/60*duration))
    if hr <140: val = val-1
    scaling_factors = generate_increasing_amplitude_wave_array(val, rr_step)
    rsa = np.tile(wave, len(scaling_factors)) * np.repeat(scaling_factors, len(wave))
    wave_f = np.tile(rsa,reps)

    ### For RR effect comment this
    # reps = int(hr)
    # wave_f = np.tile(wave, reps)

    ## Resample to intended duration
    new_length = duration*samples
    wave = np.interp(np.linspace(0, len(wave_f)-1, new_length), np.arange(len(wave_f)), wave_f)
    
    ## Select Normalization
    wave = min_val + ((wave - np.min(wave)) / (np.max(wave) - np.min(wave))) * (max_val - min_val)
    
    wave = abs(wave)
    return wave


def main(hr, rr, rr_step, max_amp, min_amp, waveform, minute, duration=300, samples=410):
    freq = hr/60
    delay_req = 1/(samples)

    i2c = busio.I2C(board.SCL, board.SDA)
    dac = a.MCP4725(i2c, address=0x60)

    try:
        while(minute>0):
            if waveform == "pulse":
                wave = pulse_gen_with_rr(min_amp, max_amp, samples, duration, hr, rr, rr_step)
            elif waveform == "sine":
                wave = sine_gen_with_rr_v4(min_amp, max_amp, samples, duration, hr, rr, rr_step)
            
            start_time = time.time()
            print('Start time:', start_time)
            for i in range(0,len(wave)-1):
                val = int(wave[i])
                dac.raw_value = val
                # delay = delay_req - 0.00041 - 0.00025   
                delay = delay_req - 0.00041 - 0.00025 - 0.000035
                time.sleep(delay)
 
            end_time = time.time()
            # print('End time:', end_time)
            
            ## Write Labels for 10s, each label after 1s
            hr_array = np.repeat(hr, duration)
            rr_array = np.repeat(rr, duration)
            # write_mqtt(hr_array, rr_array, start_time, 1)

            final_time = time.time()
            print('Final time:', final_time)
            total_time = (end_time - start_time)
            
            total_cycles = hr/60 * duration
            frequ = total_cycles/total_time

            calc_hr = 60 * frequ
            print(f"Calculated HR: {calc_hr:.2f} bpm")
            print('hr:', hr, "rr:", rr)
            minute -=1
            
    except KeyboardInterrupt:
        print('End')


if __name__== '__main__':
    
    """
    option 1, HR 40, RR 8
    option 2, HR 64, RR 16
    option 3, HR 96, RR 24
    option 4, HR 128, RR 32
    option 5, HR 160, RR 40
    """
    parser = argparse.ArgumentParser(description='Heartbeat Simulator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--minute', type=int, default=1, help='Length of Working (Unit: min), default=3')
    args = parser.parse_args()

    for option in range(1, 6):

        if option == 1:
            hr, rr, rr_step = 40, 8, 0.01
            max_amp, min_amp = 200, 0
            waveform = 'sine'
        elif option == 2:
            hr, rr, rr_step = 64, 16, 0.02
            max_amp, min_amp =  256, 0
            waveform = 'sine'
        elif option == 3:
            hr, rr, rr_step = 96, 24, 0.04
            max_amp, min_amp =  256, 0
            waveform = 'sine'   
        elif option == 4:
            hr, rr, rr_step = 128, 32, 0.04
            max_amp, min_amp =  256, 0
            waveform = 'pulse'  
        elif option == 5:
            hr, rr, rr_step = 160, 40, 0.04
            max_amp, min_amp =  512, 0
            waveform = 'pulse'  
        
        # print(get_mac())
        main(hr, rr, rr_step, max_amp, min_amp, waveform, args.minute)
