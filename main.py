import pyaudio
import numpy as np
import csv
import librosa
from matplotlib import pyplot
from matplotlib import animation
import matplotlib
from matplotlib.widgets import Button

matplotlib.use('TkAgg')

WIDTH = 2
CHANNELS = 1
RATE = 8000
BLOCKLEN = 512
DURATION = 10  # seconds
FRAME_RATE = int(RATE / BLOCKLEN)
NUM_FRAMES = DURATION * FRAME_RATE

# 全局状态变量
capturing = False
frame_count = 0

# 初始化特征存储
features = {
    'time': [],
    'pitch': [],
    'spectral_centroid': [],
    'mfccs': [],
    'bpm': None
}

# 打开音频流
p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(WIDTH)
stream = p.open(
    format=PA_FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=False,
    frames_per_buffer=BLOCKLEN
)

# 设置绘图
my_fig, (ax1, ax2) = pyplot.subplots(2, 1)
my_fig.canvas.manager.set_window_title("Intelligent Music Recommendation System based on LLM")
my_fig.set_facecolor('Gainsboro')
t = np.arange(BLOCKLEN) * (1000 / RATE)
x = np.zeros(BLOCKLEN)
X = np.fft.rfft(x)
f_X = np.arange(X.size) * RATE / BLOCKLEN

[g1] = ax1.plot([], [], color='purple')
ax1.set_xlim(0, 1000 * BLOCKLEN / RATE)
ax1.set_ylim(-20000, 20000)
ax1.set_xlabel('Time (msec)')
ax1.set_title('Real-time Waveform of Input Music')

[g2] = ax2.plot([], [], color='deeppink')
ax2.set_xlim(0, RATE / 2)
ax2.set_ylim(0, 3000)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_title('Frequency Spectrum')
my_fig.tight_layout()

my_fig.subplots_adjust(top=0.8)

n_fft = BLOCKLEN

# 在顶部添加按钮
button_ax = pyplot.axes([0.42, 0.92, 0.22, 0.05])  # 调整按钮位置和大小
# start_button = Button(button_ax, 'Start Recommend')
start_button = Button(button_ax, 'Start Recommend', color='lightblue', hovercolor='lightgreen')


def my_init():
    g1.set_xdata(t)
    g2.set_xdata(f_X)
    return g1, g2


def my_update(frame_idx):
    global capturing, frame_count, features

    if not capturing:
        # 不在捕获状态时不更新数据，只返回空值
        return g1, g2

    if frame_count >= NUM_FRAMES:
        # 已经录满10秒数据，结束捕获
        capturing = False

        # 计算BPM
        recorded_signal = np.array(features['mfccs']).flatten()
        features['bpm'] = compute_bpm(recorded_signal, RATE)

        # 写入CSV
        save_features_to_csv(features, 'audio_features.csv')

        # 停止动画事件源
        my_anima.event_source.stop()
        return g1, g2

    # 读取音频数据
    signal_bytes = stream.read(BLOCKLEN, exception_on_overflow=False)
    signal_block = np.frombuffer(signal_bytes, dtype='int16')
    signal_block = np.pad(signal_block, (0, max(0, BLOCKLEN - len(signal_block))), mode='constant')
    X_input = np.fft.rfft(signal_block) / BLOCKLEN

    # 提取特征
    pitch = librosa.piptrack(
        y=signal_block.astype(float), sr=RATE, n_fft=n_fft
    )[0].mean()
    spectral_centroid = librosa.feature.spectral_centroid(
        y=signal_block.astype(float), sr=RATE, n_fft=n_fft
    )[0].mean()
    mfccs = librosa.feature.mfcc(
        y=signal_block.astype(float), sr=RATE, n_mfcc=13, n_fft=n_fft
    ).mean(axis=1)

    # 更新特征存储
    features['time'].append(frame_count / FRAME_RATE)
    features['pitch'].append(pitch)
    features['spectral_centroid'].append(spectral_centroid)
    features['mfccs'].append(mfccs.tolist())

    # 更新绘图
    g1.set_ydata(signal_block)
    g2.set_ydata(np.abs(X_input))

    frame_count += 1
    return g1, g2


def compute_bpm(audio_signal, sr):
    tempo, _ = librosa.beat.beat_track(y=audio_signal, sr=sr)
    return tempo


def save_features_to_csv(features, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Time', 'Pitch', 'Spectral Centroid'] + [f'MFCC_{i+1}' for i in range(13)] + ['BPM']
        writer.writerow(header)
        for i in range(len(features['time'])):
            row = [
                features['time'][i],
                features['pitch'][i],
                features['spectral_centroid'][i]
            ] + features['mfccs'][i] + [features['bpm']]
            writer.writerow(row)
    print("* Finished and saved audio features to 'audio_features.csv'")


def start_capture(event):
    global capturing, frame_count, features, my_anima

    # 清空特征数据
    features = {
        'time': [],
        'pitch': [],
        'spectral_centroid': [],
        'mfccs': [],
        'bpm': None
    }

    frame_count = 0
    capturing = True

    # 每次点击按钮都重新启动动画
    my_anima = animation.FuncAnimation(
        my_fig,
        my_update,
        init_func=my_init,
        interval=1000 / FRAME_RATE,
        blit=True,
        cache_frame_data=False,
        repeat=False
    )

    # 动画开始运行
    my_anima.event_source.start()


start_button.on_clicked(start_capture)

pyplot.show()

# 程序不再在这里关闭音频流, 保持打开状态
stream.stop_stream()
stream.close()
p.terminate()
