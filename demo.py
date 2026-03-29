import whisper

model = whisper.load_model("large")
result = model.transcribe(
    "audio2.m4a",
    language="ne",
    temperature=0,
    beam_size=5,
    best_of=5
)
print(result["text"])

