import pyaudio
import audioop
import pygame
import threading

# 오디오 스트리밍을 위한 설정
CHUNK = 1024  # 한 번에 읽을 프레임 수
FORMAT = pyaudio.paInt16  # 16비트 정수 형식
CHANNELS = 1  # mono
RATE = 44100  # 샘플링 레이트 (Hz)

# 소리 임계값 설정 (dB)
THRESHOLD = 7000

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
        data = stream.read(CHUNK)
        rms = audioop.rms(data, 2)  # 16비트 mono 데이터의 RMS 계산

        if rms > THRESHOLD:
            play_sound_thread('./drum_sample/Bass-Drum-1.wav')
except KeyboardInterrupt:
    print("종료합니다.")

# 스트림 정리 및 종료
stream.stop_stream()
stream.close()
p.terminate()