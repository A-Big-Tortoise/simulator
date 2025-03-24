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



def main(max_amp, min_amp, duration=180, samples=410):
    delay_req = 1/(samples)

    i2c = busio.I2C(board.SCL, board.SDA)
    dac = a.MCP4725(i2c, address=0x60)

    try:
        # wave = np.concatenate([np.zeros(samples * 3), np.ones(samples * 3)])
        # wave = np.concatenate([np.ones(samples * 3), np.zeros(samples * 3)])
        #wave = min_amp + ((wave - np.min(wave)) / (np.max(wave) - np.min(wave))) * (max_amp - min_amp)
        wave = np.ones(samples * duration)
        wave = max_amp * wave
        start_time = time.time()
        print('Start time:', start_time)
        for i in range(0,len(wave)-1):
            val = int(wave[i])
            dac.raw_value = val
            # delay = delay_req - 0.00041 - 0.00025   
            delay = delay_req - 0.00041 - 0.00025 - 0.000035
            time.sleep(delay)

        end_time = time.time()
        print('Final time:', end_time)
        total_time = (end_time - start_time)
                    
    except KeyboardInterrupt:
        print('End')


if __name__== '__main__':

    parser = argparse.ArgumentParser(description='Heartbeat Simulator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_amp', type=int, help='Max Amp')
    args = parser.parse_args()



        
    # max_amp, min_amp =  64, 0 

    main(args.max_amp, 0)
