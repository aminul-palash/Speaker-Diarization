uvicorn app:app --host 0.0.0.0 --port 8000

curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@sample.wav" \
  -F "rttm=@sample.rttm"