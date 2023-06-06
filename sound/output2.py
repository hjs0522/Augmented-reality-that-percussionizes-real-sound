import pyaudio
import wave
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paInt16  # 16-bit format
CHANNELS = 1  # mono
RATE = 44100  # samples per second
RECORD_SECONDS = 5  # recording duration
WAVE_OUTPUT_FILENAME = "output2.wav"
THRESHOLD = 5000  # Amplitude threshold

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []
recording = False

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    npdata = np.frombuffer(data, dtype=np.int16)
    if np.any(np.abs(npdata) > THRESHOLD):
        frames.append(data)
        recording = True
    elif recording:
        print("* done recording")
        break

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
samplerate, data = wavfile.read('output2.wav')

# Create a time array in seconds
times = np.arange(len(data)) / float(samplerate)

# Plot the sound wave
plt.figure(figsize=(20, 4))
plt.fill_between(times, data, color='k')  # data[:,0] if the file is stereo
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
plt.show()