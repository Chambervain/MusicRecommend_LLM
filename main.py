import pyaudio
import numpy as np
import csv
import librosa
from matplotlib import pyplot
from matplotlib import animation
import matplotlib

matplotlib.use('TkAgg')
WIDTH = 2
CHANNELS = 1
RATE = 8000
BLOCKLEN = 512
DURATION = 10  # seconds
FRAME_RATE = int(RATE / BLOCKLEN)
NUM_FRAMES = DURATION * FRAME_RATE

# Initialize feature storage
features = {
    'time': [],
    'pitch': [],
    'spectral_centroid': [],
    'mfccs': [],
    'bpm': None
}

# Open audio stream
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

# Set up plotting
my_fig, (ax1, ax2) = pyplot.subplots(2, 1)
my_fig.canvas.manager.set_window_title("Intelligent Music Recommendation System based on LLM")
t = np.arange(BLOCKLEN) * (1000 / RATE)
x = np.zeros(BLOCKLEN)
X = np.fft.rfft(x)
f_X = np.arange(X.size) * RATE / BLOCKLEN

[g1] = ax1.plot([], [])
ax1.set_xlim(0, 1000 * BLOCKLEN / RATE)
ax1.set_ylim(-10000, 10000)
ax1.set_xlabel('Time (msec)')
ax1.set_title('Real-time Waveform of Input Music')

[g2] = ax2.plot([], [])
ax2.set_xlim(0, RATE / 2)
ax2.set_ylim(0, 1000)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_title('Frequency Spectrum')
my_fig.tight_layout()

n_fft = BLOCKLEN


def my_init():
    g1.set_xdata(t)
    g2.set_xdata(f_X)
    return g1, g2


def my_update(frame_idx):
    if frame_idx >= NUM_FRAMES:
        pyplot.close()
        return []

    # Read audio input stream
    signal_bytes = stream.read(BLOCKLEN, exception_on_overflow=False)
    signal_block = np.frombuffer(signal_bytes, dtype='int16')

    signal_block = np.pad(signal_block, (0, max(0, BLOCKLEN - len(signal_block))), mode='constant')
    X_input = np.fft.rfft(signal_block) / BLOCKLEN

    # Extract features
    pitch = librosa.piptrack(
        y=signal_block.astype(float), sr=RATE, n_fft=n_fft
    )[0].mean()
    spectral_centroid = librosa.feature.spectral_centroid(
        y=signal_block.astype(float), sr=RATE, n_fft=n_fft
    )[0].mean()
    mfccs = librosa.feature.mfcc(
        y=signal_block.astype(float), sr=RATE, n_mfcc=13, n_fft=n_fft
    ).mean(axis=1)

    # Update feature storage
    features['time'].append(frame_idx / FRAME_RATE)
    features['pitch'].append(pitch)
    features['spectral_centroid'].append(spectral_centroid)
    features['mfccs'].append(mfccs.tolist())

    # Update plots
    g1.set_ydata(signal_block)
    g2.set_ydata(np.abs(X_input))
    return g1, g2


def compute_bpm(audio_signal, sr):
    tempo, _ = librosa.beat.beat_track(y=audio_signal, sr=sr)
    return tempo

my_anima = animation.FuncAnimation(
    my_fig,
    my_update,
    init_func=my_init,
    interval=1000 / FRAME_RATE,
    blit=True,
    cache_frame_data=False,
    repeat=False
)

recorded_signal = []
pyplot.show()

stream.stop_stream()
stream.close()
p.terminate()

recorded_signal = np.array(features['mfccs']).flatten()
features['bpm'] = compute_bpm(recorded_signal, RATE)

# Save features to a CSV file
with open('audio_features.csv', 'w', newline='') as f:
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
