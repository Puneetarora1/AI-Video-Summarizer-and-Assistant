import yt_dlp
from pydub import AudioSegment # for chunking of audio
import os

"""
Why WAV is used:

It's uncompressed and universally supported. Almost every audio library and speech tool can read WAV without extra codecs.
It's lossless. Converting from .webm or .m4a to WAV doesn't lose any more quality, whereas converting to mp3 would compress it a
second time.
Speech models work on raw audio anyway. If this audio goes into Whisper for transcription, Whisper decodes whatever file you
give it through ffmpeg and resamples it to 16 kHz mono internally. A WAV is already in that decoded form, so it avoids surprises 
with odd codecs or containers.
"""

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url : str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",  # downloads the best audio stream and falls back to the best overall format if there's no separate audio.
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio", # postprocessor that extracts or converts audio, the same one -x uses. It requires ffmpeg and ffprobe to be installed.
                "preferredcodec": "wav", # target format
                "preferredquality": "192", # quality settings, WAV is lossless, so it has no effect here.
            }
        ],
        "quiet": True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename


def convert_to_wav(input_path: str) -> str:
    "Convert any audio/video file to WAVE format using pydub"
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)                          # decides which type of file is coming
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path : str, chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    
    chunks = []
    
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format ="wav")
        
        chunks.append(chunk_path)
        
    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected Youtube URL, Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file, Converting to WAV...")
        wav_path = convert_to_wav(source)
    
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks