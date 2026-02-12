import numpy as np
import soundfile as sf
import os

sample_rate = 16000
duration = 5 # seconds
frequency = 440 # Hz

t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio = 0.5 * np.sin(2 * np.pi * frequency * t)

sf.write('sample.wav', audio, sample_rate)
print("Created sample.wav")
