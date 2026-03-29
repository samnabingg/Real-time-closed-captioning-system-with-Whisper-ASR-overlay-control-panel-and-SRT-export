import sounddevice as sd
import numpy as np
import queue
import collections
import threading

print(sd.query_devices())

WHISPER_SAMPLE_RATE = 16000
PREFERRED_DEVICES = [1, 5, 9]


def find_working_device():
    for device_index in PREFERRED_DEVICES:
        try:
            device_info = sd.query_devices(device_index, 'input')
            sample_rate = int(device_info['default_samplerate'])
            test_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                device=device_index,
                blocksize=int(sample_rate * 0.1)
            )
            test_stream.close()
            print(f"Using device {device_index}: {device_info['name']} @ {sample_rate}Hz")
            return device_index, sample_rate
        except Exception as e:
            print(f"Device {device_index} failed: {e}")
            continue
    raise RuntimeError("No working input device found.")


DEVICE_INDEX, DEVICE_SAMPLE_RATE = find_working_device()

CHUNK_DURATION = 0.1
CHUNK_SIZE = int(DEVICE_SAMPLE_RATE * CHUNK_DURATION)

# Thread-safe queue — audio callback deposits raw chunks here
audio_queue = queue.Queue()

# Rolling ring buffer — stores (chunk_array, wall_clock_time) tuples
# Wall clock time is used to track absolute position across cycles
RING_BUFFER_SECONDS = 30
ring_buffer = collections.deque(
    maxlen=int(RING_BUFFER_SECONDS / CHUNK_DURATION)
)
ring_buffer_lock = threading.Lock()


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