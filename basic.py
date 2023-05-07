import pyaudio
import numpy as np
import time


# 오디오 설정 변수
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# 이퀄라이저 게인 설정
low_gain = 1.0
mid_gain = 1.5
high_gain = 2.0

def equalizer(data, low_gain, mid_gain, high_gain):
    # 오디오 데이터를 numpy 배열로 변환
    audio_data = np.frombuffer(data, dtype=np.int16)

    # 고속 푸리에 변환 (FFT)을 사용하여 주파수 영역으로 변환
    freq_data = np.fft.rfft(audio_data)

    # 주파수 영역에서 각 대역의 게인 조정
    freq_data[0:int(CHUNK/8)] *= low_gain
    freq_data[int(CHUNK/8):int(CHUNK/4)] *= mid_gain
    freq_data[int(CHUNK/4):] *= high_gain

    # 역 고속 푸리에 변환 (IFFT)을 사용하여 시간 영역으로 변환
    audio_data = np.fft.irfft(freq_data)

    # numpy 배열을 바이트 형식으로 변환
    processed_data = audio_data.astype(np.int16).tobytes()

    return processed_data

# PyAudio 객체 생성
audio = pyaudio.PyAudio()

# 오디오 스트림 열기 (입력 및 출력)
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

print("소리를 입력하세요! 종료하려면 Ctrl+C를 누르세요.")

try:
    while True:
        # 입력 스트림에서 오디오 데이터 읽기
        data = stream.read(CHUNK)

        # 이퀄라이저 적용
        processed_data = equalizer(data, low_gain, mid_gain, high_gain)

        # 출력 스트림에 오디오 데이터 쓰기 (재생)
        stream.write(processed_data)
except KeyboardInterrupt:
    print("종료합니다.")

# 스트림 정리 및 종료
stream.stop_stream()
stream.close()
audio.terminate()