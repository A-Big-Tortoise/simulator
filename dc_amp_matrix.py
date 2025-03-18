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


def reset():
    "Use 2 sec to reset the motor, make the motor hit the button"
    reset_wave = np.ones(samples * 2)
    for i in range(0,len(reset_wave)-1):
        dac.raw_value = int(reset_wave[i])
        delay = delay_req - 0.00041 - 0.00025 - 0.000035
        time.sleep(delay)
    
def square_wave(amp, samples, duty_cycle):
    high_length = int(samples * duty_cycle)
    low_length = samples - high_length
    high_part = np.full(high_length, amp)
    low_part = np.zeros(low_length)
    return np.concatenate([high_part, low_part])


def sine_wave(amp, samples, duty_cycle):
    f_hr = 0.5 / duty_cycle
    t = np.linspace(0, 1, samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * f_hr * t)[:int(samples * duty_cycle)]
    zeros = np.zeros(samples - int(samples * duty_cycle))
    return np.concatenate([sine_wave, zeros])


def main(max_amp, min_amp, duty_circle, waveform, duration=180, samples=410):
    delay_req = 1/(samples)

    i2c = busio.I2C(board.SCL, board.SDA)
    dac = a.MCP4725(i2c, address=0x60)

    reset()


    wave = np.ones(samples * 5)
    wave = max_amp * wave
    start_time = time.time()
    print('Start time:', start_time)
    for i in range(0,len(wave)-1):
        val = int(wave[i])
        dac.raw_value = val 
        delay = delay_req - 0.00041 - 0.00025 - 0.000035
        time.sleep(delay)

    end_time = time.time()
    print('Final time:', end_time)
    print(f'Total time: {end_time - start_time}')
                


if __name__== '__main__':

    parser = argparse.ArgumentParser(description='Heartbeat Simulator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--waveform', type=str, help='Waveform')
    args = parser.parse_args()


    max_amps = [4096, 2048, 1024, 512, 256, 128]
    duty_circles = [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


    for dc in duty_circles:
        for max_amp in max_amps:
        main(max_amp, 0, dc, args.waveform)
