import sounddevice as sd
import numpy as np
import queue

print(sd.query_devices())

WHISPER_SAMPLE_RATE = 16000

# Device preference order — tries each one until one works
# MME (1) is most compatible on Windows, DirectSound (5) is next
PREFERRED_DEVICES = [1, 5, 9]

def find_working_device():
    """Try each preferred device and return the first one that opens successfully."""
    for device_index in PREFERRED_DEVICES:
        try:
            device_info = sd.query_devices(device_index, 'input')
            sample_rate = int(device_info['default_samplerate'])

            # Try opening a test stream to confirm it works
            test_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                device=device_index,
                blocksize=int(sample_rate * 1.0)
            )
            test_stream.close()

            print(f"Using device {device_index}: {device_info['name']} @ {sample_rate}Hz")
            return device_index, sample_rate

        except Exception as e:
            print(f"Device {device_index} failed: {e}")
            continue

    raise RuntimeError("No working input device found. Check your microphone connections.")


DEVICE_INDEX, DEVICE_SAMPLE_RATE = find_working_device()
CHUNK_SIZE = int(DEVICE_SAMPLE_RATE * 1.0)  # 1 second chunks

audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"[Audio status]: {status}")
    audio_queue.put(indata.copy())

stream = sd.InputStream(
    samplerate=DEVICE_SAMPLE_RATE,
    channels=1,
    callback=audio_callback,
    dtype='float32',
    blocksize=CHUNK_SIZE,
    device=DEVICE_INDEX
)