# Real-Time Captioning Latency Reduction
Sweet spot: ~2-4s latency with good accuracy.

## Steps (in order):

### 1. Update Parameters (RLT/whisper_worker.py) ✓
- VAD_CHECK_INTERVAL: 0.25s (from 0.5s)
- MAX_SEGMENT_SECONDS: 5s (from 15s) 
- MIN_SPEECH_SECONDS: 0.5s (from 0.3s)
- PADDING_SECONDS: 0.3s (from 0.5s)
- Whisper: beam_size=3, best_of=3, condition_on_previous_text=True ✓

### 2. Reduce Ring Buffer (RLT/audio_streaming.py) ✓
- RING_BUFFER_SECONDS: 8s (from 30s)

### 3. Test Short Utterances ✓
- Latency still ~10s (Whisper bottleneck). Good: buffer/VAD faster.

### 4. Test Long Speech
- Speak 10-20s continuously, verify streaming (no 15s waits).

### 5. Monitor & Fine-tune
- Latencies ~9-10s → Switch to 'tiny' model.
- Further: beam_size=1 if needed.

### 6. Switch to 'tiny' model (Next)
- Edit whisper_worker.py model='tiny'
### 7. Complete ✓
