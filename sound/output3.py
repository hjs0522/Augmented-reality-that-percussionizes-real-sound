import pyaudio
import wave
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16  # 16-bit format
CHANNELS = 1  # mono
RATE = 44100  # samples per second
THRESHOLD = 5000  # Amplitude threshold
WAVE_OUTPUT_FILENAME = "output3.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []
recording = False

while True:
    data = stream.read(CHUNK)
    npdata = np.frombuffer(data, dtype=np.int16)
    
    # If the audio data crosses the threshold and we're not already recording, start recording
    if np.any(np.abs(npdata) > THRESHOLD) and not recording:
        print("Recording started")
        recording = True
        
    # If we're recording and the audio data goes below the threshold, stop recording
    elif recording and np.all(np.abs(npdata) <= THRESHOLD):
        print("Recording stopped")
        recording = False
        break
    
    if recording:
        frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Read the wav file
samplerate, data = wavfile.read('output3.wav')

# Create a time array in seconds
times = np.arange(len(data)) / float(samplerate)

# Plot the sound wave
plt.figure(figsize=(20, 4))
plt.fill_between(times, data, color='k')  # data[:,0] if the file is stereo
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.show()
