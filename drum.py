import pyaudio
import audioop
import pygame
import threading
import numpy as np
import time
import os
import librosa
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def load_samples(directory):
    samples = {}
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            sample_name = os.path.splitext(file)[0]
            file_path = os.path.join(directory, file)
            samples[sample_name], _ = librosa.load(file_path, sr=44100)
    return samples

samples_directory = '/Users/junseobhong/capstone/drum_sample'
drum_samples = load_samples(samples_directory)

# 각 샘플에 대한 파일 경로를 저장합니다.
drum_samples_paths = {name: os.path.join(samples_directory, name + '.wav') for name in drum_samples.keys()}

drum_mfcc_dic = {}
# 드럼 소리의 MFCC 계산
for key,value in drum_samples.items():
    drum_mfcc = librosa.feature.mfcc(y=value, sr=44100)
    drum_mfcc_mean = drum_mfcc.mean(axis=1)
    drum_mfcc_dic[key] = drum_mfcc_mean

#가장 비슷한 드럼 소리를 찾아주는 함수
def find_nearest_drum_sound(input_audio):
    # 입력 오디오의 MFCC 계산
    """
    mfcc 를 통해 소리의 특징을 분석할 수 있다. 그렇기에 입력 소리의 mfcc를 추출하여 사용한다.
    """
    input_mfcc = librosa.feature.mfcc(y=input_audio, sr=44100)
    input_mfcc_mean = input_mfcc.mean(axis=1)
    print(input_mfcc_mean)
    min_distance = float('inf')
    nearest_drum = None
    
    """
    mfcc간의 비교를 하는 경우 가장 많이 쓰는 유클리디안 거리를 사용하여 입력 오디오와 드럽 소리의 유사성을 계산한다.
    """
    for key,value in drum_samples.items():
        # 유클리디안 거리를 사용하여 입력 오디오와 드럼 소리의 유사성 계산
        distance = euclidean(input_mfcc_mean, drum_mfcc_dic[key])
        print(key,distance)
        # 가장 유사한 드럼 소리 찾기
        if distance < min_distance:
            min_distance = distance
            nearest_drum = key
    return nearest_drum


# 오디오 스트리밍을 위한 설정
CHUNK = 1024*2  # 한 번에 읽을 프레임 수
FORMAT = pyaudio.paInt16  # 16비트 정수 형식
CHANNELS = 1  # mono
RATE = 44100  # 샘플링 레이트 (Hz)

# 소리 임계값 설정 (dB)
THRESHOLD = 5000

# pyaudio 인스턴스 생성
p = pyaudio.PyAudio()

# 입력 스트림 생성
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

def play_sound(file):
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # 음악 재생이 끝나기를 기다림
        continue

# 스레드를 사용하여 오디오 재생
def play_sound_thread(file):
    t = threading.Thread(target=play_sound, args=(file,))
    t.start()

# 스트림을 지속적으로 읽고 소리 크기 확인
try:
    while True:
        start_time = time.time()
        data = stream.read(CHUNK,exception_on_overflow = False)
        # 바이트 형식의 입력 오디오 데이터를 numpy 배열로 변환 => 계산하기 쉬워짐
        input_audio = np.frombuffer(data, dtype=np.int16)

        # 입력 오디오 데이터를 부동 소수점으로 변환
        input_audio = input_audio.astype(np.float32) / 32767.0

        rms = audioop.rms(data, 2)  # 16비트 mono 데이터의 RMS 계산
        """
        만약 THRESHOLD를 설정해 주지 않는다면 stream.read는 계속 동작 하고있기 때문에 아무 소리가 입력되지 않을때도
        입력 소리와 가장 비슷한 소리를 찾고 출력해주게 된다. 그렇기에 THRESHOLD를 이용해 특정 크기 이상의 소리가 입력되었을 경우만
        가장 비슷한 소리를 찾고 출력되도록 한다.
        """
        if rms > THRESHOLD:
            # 가장 유사한 드럼 소리 찾기
            nearest_drum = find_nearest_drum_sound(input_audio)
            play_sound_thread(drum_samples_paths[nearest_drum])
            # 코드 실행 후 시간을 측정합니다.
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"코드 실행 시간: {elapsed_time:.5f} 초")
             # 그래프 그리기
            plt.figure(figsize=(14, 5))
            plt.plot(input_audio)
            plt.show()
            print(nearest_drum)
except KeyboardInterrupt:
    print("종료합니다.")

# 스트림 정리 및 종료
stream.stop_stream()
stream.close()
p.terminate()