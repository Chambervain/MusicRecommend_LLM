import pyaudio
import matplotlib
from matplotlib import pyplot
from matplotlib import animation
import numpy as np

matplotlib.use('TkAgg')
WIDTH = 2
CHANNELS = 1
RATE = 8000
BLOCKLEN = 512


# Open the audio stream
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

# Set up plotting with two plots (2x1 grid)
my_fig, (ax1, ax2) = pyplot.subplots(2, 1)
my_fig.canvas.manager.set_window_title("Intelligent Music Recommendation System based on LLM")
t = np.arange(BLOCKLEN) * (1000 / RATE)
x = np.zeros(BLOCKLEN)
X = np.fft.rfft(x)
f_X = np.arange(X.size) * RATE / BLOCKLEN

# Input signal plot
[g1] = ax1.plot([], [])
ax1.set_xlim(0, 1000 * BLOCKLEN / RATE)
ax1.set_ylim(-10000, 10000)
ax1.set_xlabel('Time (msec)')
ax1.set_title('Real-time Waveform of Input Music')

# Fourier Transform of input signal
[g2] = ax2.plot([], [])
ax2.set_xlim(0, RATE / 2)
ax2.set_ylim(0, 1000)
ax2.set_xlabel('Frequency (Hz)')
ax2.set_title('Frequency Spectrum')
my_fig.tight_layout()

# Define animation functions
def my_init():
    g1.set_xdata(t)
    g2.set_xdata(f_X)
    return (g1, g2)

def my_update(i):
    # Read audio input stream
    signal_bytes = stream.read(BLOCKLEN, exception_on_overflow=False)
    signal_block = np.frombuffer(signal_bytes, dtype='int16')

    # Compute frequency spectrum for input signal
    X_input = np.fft.rfft(signal_block) / BLOCKLEN

    # Update the plots
    g1.set_ydata(signal_block)
    g2.set_ydata(np.abs(X_input))
    return (g1, g2)

# Read microphone, plot audio signal and its frequency spectrum
my_anima = animation.FuncAnimation(
    my_fig,
    my_update,
    init_func=my_init,
    interval=10,
    blit=True,
    cache_frame_data=False,
    repeat=False
)

pyplot.show()
stream.stop_stream()
stream.close()
p.terminate()

print('* Finished')
