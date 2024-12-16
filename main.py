import pyaudio
import numpy as np
import librosa
from matplotlib import pyplot
from matplotlib import animation
import matplotlib
from matplotlib.widgets import Button
import openai

OPENAI_API_KEY = "MY API KEY"
openai.api_key = OPENAI_API_KEY

matplotlib.use('TkAgg')
WIDTH = 2
CHANNELS = 1
RATE = 8000
BLOCKLEN = 512
DURATION = 10  # seconds
FRAME_RATE = int(RATE / BLOCKLEN)
NUM_FRAMES = DURATION * FRAME_RATE
capturing = False
frame_count = 0

features = {
    'time': [],
    'pitch': [],
    'spectral_centroid': [],
    'mfccs': [],
    'bpm': None
}

all_audio_blocks = []
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
button_ax = pyplot.axes([0.42, 0.92, 0.22, 0.05])
start_button = Button(button_ax, 'Start Recommend', color='lightblue', hovercolor='lightgreen')

my_anima = None


def my_init():
    g1.set_xdata(t)
    g2.set_xdata(f_X)
    return g1, g2


def compute_bpm(audio_signal, sr):
    tempo, _ = librosa.beat.beat_track(y=audio_signal.astype(float), sr=sr)
    return tempo


def get_song_recommendations(description):
    prompt = (
        f"Based on the following music description, recommend 3 songs with a similar style:\n\n"
        f"{description}\n\n"
        "Please provide the name of the songs, their singers, and the album name in a list format."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a music expert."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


def my_update(frame_idx):
    global capturing, frame_count, features, all_audio_blocks, my_anima

    if not capturing:
        return g1, g2

    if frame_count >= NUM_FRAMES:
        # 停止录制
        capturing = False
        recorded_signal = np.concatenate(all_audio_blocks)
        features['bpm'] = compute_bpm(recorded_signal, RATE)

        pitch_array = np.array(features['pitch'])
        centroid_array = np.array(features['spectral_centroid'])
        mfccs_array = np.array(features['mfccs'])  # shape: [num_frames, 13]

        pitch_var = np.mean(pitch_array)
        centroid_var = np.mean(centroid_array)
        mfccs_var = np.mean(mfccs_array, axis=0)
        bpm_var = features['bpm'][0]

        print("Pitch Average:", pitch_var)
        print("Spectral Centroid Average:", centroid_var)
        print("MFCCs Average:", mfccs_var)
        print("BPM:", bpm_var)

        # 根据bpm描述节奏
        # if bpm_var < 60:
        #     tempo_desc = "a slow and soothing tempo"
        # elif 60 <= bpm_var <= 90:
        #     tempo_desc = "a moderate tempo"
        # elif 90 < bpm_var <= 150:
        #     tempo_desc = "a lively and energetic tempo"
        # else:
        #     tempo_desc = "a very fast tempo"
        tempo_desc = "average BPM is " + str(bpm_var)

        # 根据频谱质心描述音色明亮度
        # if centroid_var < 2000:
        #     tone_desc = "a warm and smooth sound"
        # else:
        #     tone_desc = "a bright and crisp sound"
        tone_desc = "average spectral centroid is " + str(centroid_var)

        # 根据基频描述音高
        # if pitch_var < 300:
        #     pitch_desc = "with a low-pitched melody"
        # else:
        #     pitch_desc = "with a high-pitched melody"
        pitch_desc = "average pitch is " + str(pitch_var)

        description = (f"The audio features of input music are: {tempo_desc}, {tone_desc}, {pitch_desc}.")
        print("Generated real-time input music description:")
        print(description)

        # 调用大语言模型API获取推荐歌曲
        recommendations = get_song_recommendations(description)
        print("Recommended Songs:")
        print(recommendations)

        if my_anima is not None:
            my_anima.event_source.stop()
        return g1, g2

    signal_bytes = stream.read(BLOCKLEN, exception_on_overflow=False)
    signal_block = np.frombuffer(signal_bytes, dtype='int16')
    signal_block = np.pad(signal_block, (0, max(0, BLOCKLEN - len(signal_block))), mode='constant')
    all_audio_blocks.append(signal_block)
    X_input = np.fft.rfft(signal_block) / BLOCKLEN

    # 提取特征
    pitch = librosa.piptrack(y=signal_block.astype(float), sr=RATE, n_fft=n_fft)[0].mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=signal_block.astype(float), sr=RATE, n_fft=n_fft)[0].mean()
    mfccs = librosa.feature.mfcc(y=signal_block.astype(float), sr=RATE, n_mfcc=13, n_fft=n_fft).mean(axis=1)

    features['time'].append(frame_count / FRAME_RATE)
    features['pitch'].append(pitch)
    features['spectral_centroid'].append(spectral_centroid)
    features['mfccs'].append(mfccs.tolist())

    g1.set_ydata(signal_block)
    g2.set_ydata(np.abs(X_input))
    frame_count += 1
    return g1, g2


def start_capture(event):
    global capturing, frame_count, features, my_anima, all_audio_blocks

    features = {
        'time': [],
        'pitch': [],
        'spectral_centroid': [],
        'mfccs': [],
        'bpm': None
    }

    all_audio_blocks = []
    frame_count = 0
    capturing = True

    my_anima = animation.FuncAnimation(
        my_fig,
        my_update,
        init_func=my_init,
        interval=1000 / FRAME_RATE,
        blit=True,
        cache_frame_data=False,
        repeat=False
    )
    my_anima.event_source.start()


start_button.on_clicked(start_capture)
pyplot.show()
stream.stop_stream()
stream.close()
p.terminate()
