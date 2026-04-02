FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
WORKDIR /workspace

RUN apt-get update && apt-get install -y git ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod "huggingface_hub[cli]"

RUN git clone https://github.com/HeartMuLa/heartlib.git && \
    cd heartlib && \
    pip install -e .

RUN hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen' && \
    hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year' && \
    hf download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss-20260123'

COPY handler.py .
CMD ["python", "-u", "handler.py"]
