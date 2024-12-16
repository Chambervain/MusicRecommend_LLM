import pyaudio
import wave
import numpy as np
import librosa
import requests
import openai
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import threading
import time
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import hmac
import hashlib
import base64
import os
from collections import deque
matplotlib.use("TkAgg")

####################################
# Configured API keys and parameters
####################################
config = {
    'host': 'identify-us-west-2.acrcloud.com',
    'access_key': 'MY_ACCESS_KEY',
    'access_secret': 'MY_ACCESS_SECRET',
    'timeout': 10
}

openai.api_key = "MY_API_KEY"
MODEL = "gpt-4o-mini"

####################
# Parameter settings
####################
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
PLOT_UPDATE_INTERVAL = 20
MAX_SAMPLES = CHUNK * 10
downsample_factor = 8

audio_buffer = deque(maxlen=MAX_SAMPLES)
recorded_frames = []
recording_flag = False
lock = threading.Lock()

def continuous_audio_thread():
    # Continuously read audio data and update audio_buffer
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        np_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        with lock:
            audio_buffer.extend(np_data)
            if recording_flag:
                recorded_frames.append(data)


def write_wav_from_frames(frames, filename='temp.wav'):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def extract_features(audio_array, sr=RATE):
    tempo, beat_frames = librosa.beat.beat_track(y=audio_array, sr=sr)
    bpm = float(tempo[0]) if tempo.size > 0 else 0.0
    f0 = librosa.yin(audio_array, fmin=50, fmax=2000, sr=sr)
    if np.any(np.isfinite(f0)):
        pitch_val = np.median(f0[np.isfinite(f0)])
    else:
        pitch_val = 0.0
    pitch = float(pitch_val)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sr)
    spectral_centroid_val = np.mean(spectral_centroids)
    spectral_centroid = float(spectral_centroid_val)

    return bpm, pitch, spectral_centroid


def acrcloud_identify(wav_filename):
    http_method = "POST"
    http_uri = "/v1/identify"
    data_type = "audio"
    signature_version = "1"
    timestamp = str(int(time.time()))
    access_key = config['access_key']
    access_secret = config['access_secret']
    string_to_sign = "\n".join([http_method, http_uri, access_key, data_type, signature_version, timestamp])
    sign = hmac.new(access_secret.encode('utf-8'), string_to_sign.encode('utf-8'), hashlib.sha1).digest()
    signature = base64.b64encode(sign).decode('utf-8')
    sample_bytes = os.path.getsize(wav_filename)

    files = [
        ('sample', (os.path.basename(wav_filename), open(wav_filename, 'rb'), 'audio/wav'))
    ]

    data = {
        'access_key': access_key,
        'data_type': data_type,
        'signature_version': signature_version,
        'sample_bytes': sample_bytes,
        'signature': signature,
        'timestamp': timestamp
    }

    url = f"http://{config['host']}{http_uri}"
    response = requests.post(url, files=files, data=data, timeout=config['timeout'])
    try:
        result = response.json()
        if result.get('status', {}).get('code') == 0 and 'metadata' in result and 'music' in result['metadata']:
            music_info = result['metadata']['music'][0]
            title = music_info.get('title', 'Unknown')
            artist = music_info.get('artists', [{}])[0].get('name', 'Unknown')
            album = music_info.get('album', {}).get('name', 'Unknown')
            return title, artist, album
        else:
            return "Unknown", "Unknown", "Unknown"
    except:
        return "Unknown", "Unknown", "Unknown"


def get_recommendations(song_title, artist, bpm, pitch, spectral_centroid):
    prompt = f"""
You are a professional music recommendation system. I have identified the song the user is listening to as "{song_title}" by {artist}.
The song's features include:
- BPM: {bpm:.2f}
- Pitch: {pitch:.2f}
- Spectral Centroid: {spectral_centroid:.2f}

Based on the style, rhythm, and spectral characteristics of this song, please recommend 3 similar songs, including the song title, artist, and album name.
"""
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


class RealTimePlotter:
    def __init__(self, master):
        self.master = master
        self.master.title("AI Intelligent Music Recommendation & Real-Time Visualization System")
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.figure.patch.set_facecolor('#f0f0f0')

        self.ax_wave = self.figure.add_subplot(2, 1, 1)
        self.ax_wave.set_title("Waveform of Input Music", fontsize=12)
        self.ax_wave.set_facecolor('#ffffff')
        self.ax_wave.grid(True, linestyle=':', color='gray', alpha=0.7)
        self.ax_wave.set_xlim(0, MAX_SAMPLES / downsample_factor)
        self.ax_wave.set_ylim(-0.25, 0.25)
        self.wave_line, = self.ax_wave.plot([], [], color='#4B0082', linewidth=1.0)

        self.ax_spec = self.figure.add_subplot(2, 1, 2)
        self.ax_spec.set_title("Frequency Spectrum", fontsize=12)
        self.ax_spec.set_facecolor('#ffffff')
        self.ax_spec.grid(True, linestyle=':', color='gray', alpha=0.7)
        self.ax_spec.set_xlim(0, RATE / 2)
        self.ax_spec.set_ylim(1e-4, 1e3)
        self.ax_spec.set_yscale('log')
        self.spec_line, = self.ax_spec.plot([], [], color='#C44E52', linewidth=1.0)

        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.text_box = ScrolledText(master, width=60, height=10, font=("Helvetica", 10))
        self.text_box.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12))
        self.start_button = ttk.Button(master, text="Start 10s Recording and Analysis", command=self.start_10s_recording,
                                       style='TButton')
        self.start_button.pack(side=tk.BOTTOM, pady=5)
        self.updating_plot = True
        self.master.after(PLOT_UPDATE_INTERVAL, self.update_plot_continuous)

    def start_10s_recording(self):
        global recording_flag, recorded_frames
        recorded_frames.clear()
        recording_flag = True
        self.start_button.config(state=tk.DISABLED)
        self.safe_append_text("Starting the formal 10-second recording...\n")
        threading.Thread(target=self.record_10s_and_analyze).start()

    def record_10s_and_analyze(self):
        global recording_flag
        time.sleep(RECORD_SECONDS)
        # Stop recording
        recording_flag = False
        self.safe_append_text("10-second recording complete. Analyzing...\n")
        wave_data = b''.join(recorded_frames)
        audio_array = np.frombuffer(wave_data, dtype=np.int16).astype(np.float32) / 32768.0

        bpm, pitch, spectral_centroid = extract_features(audio_array)
        self.safe_append_text("Audio Features:\n"
                              f"BPM: {bpm:.2f}\n"
                              f"Pitch: {pitch:.2f}\n"
                              f"Spectral Centroid: {spectral_centroid:.2f}\n")

        temp_wav = "temp_final.wav"
        write_wav_from_frames(recorded_frames, temp_wav)
        self.safe_append_text("Identifying the song...\n")
        title, artist, album = acrcloud_identify(temp_wav)
        if title == "Unknown":
            self.safe_append_text("Failed to identify the current song.\n")
        else:
            self.safe_append_text(f"Identified Song Information:\nTitle: {title}\nArtist: {artist}\nAlbum: {album}\n")

        self.safe_append_text("Getting recommendation response based on audio features and identification...\n")
        recommendations = get_recommendations(title, artist, bpm, pitch, spectral_centroid)
        self.safe_append_text("Generated Recommended songs according to GPT LLM model:\n"+recommendations+"\n"+"\n"+"\n"+"\n")
        self.master.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def update_plot_continuous(self):
        self.update_plots()
        if self.updating_plot:
            self.master.after(PLOT_UPDATE_INTERVAL, self.update_plot_continuous)

    def update_plots(self):
        with lock:
            buffer_data = np.array(audio_buffer)
        buffer_data_ds = buffer_data[::downsample_factor]
        needed_length = int(MAX_SAMPLES / downsample_factor)
        if len(buffer_data_ds) < needed_length:
            padded = np.zeros(needed_length, dtype=buffer_data_ds.dtype)
            padded[:len(buffer_data_ds)] = buffer_data_ds
            buffer_data_ds = padded

        x_wave = np.arange(len(buffer_data_ds))
        self.wave_line.set_xdata(x_wave)
        self.wave_line.set_ydata(buffer_data_ds)
        N = len(buffer_data_ds)
        if N > 1:
            fft_data = np.fft.rfft(buffer_data_ds)
            freq = np.fft.rfftfreq(N, d=1.0 / RATE)
            magnitude = np.abs(fft_data)
            self.spec_line.set_xdata(freq)
            self.spec_line.set_ydata(magnitude)
        else:
            self.spec_line.set_xdata([])
            self.spec_line.set_ydata([])

        self.canvas.draw()

    def safe_append_text(self, text):
        self.master.after(0, self.append_text, text)

    def append_text(self, text):
        self.text_box.insert(tk.END, text)
        self.text_box.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimePlotter(root)
    # Run continuous_audio_thread when the program starts
    threading.Thread(target=continuous_audio_thread, daemon=True).start()
    root.mainloop()
