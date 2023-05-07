import pyaudio
import numpy as np
import time


# 오디오 설정 변수
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# 디스토션 설정 변수
gain = 20.0
threshold = 32767 / 2  # 클리핑 임계값을 선택 (16비트 오디오의 절반)

def distortion(audio_data, gain, threshold):
    audio_data = audio_data * gain
    audio_data = np.clip(audio_data, -threshold, threshold)
    return audio_data

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

        # 오디오 데이터를 numpy 배열로 변환
        audio_data = np.frombuffer(data, dtype=np.int16)

        # 디스토션 적용
        distorted_data = distortion(audio_data, gain, threshold)

        # numpy 배열을 바이트 형식으로 변환
        processed_data = distorted_data.astype(np.int16).tobytes()

        # 출력 스트림에 오디오 데이터 쓰기 (재생)
        stream.write(processed_data)
except KeyboardInterrupt:
    print("종료합니다.")

# 스트림 정리 및 종료
stream.stop_stream()
stream.close()
audio.terminate()
