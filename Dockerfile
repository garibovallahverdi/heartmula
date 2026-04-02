FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    runpod \
    "huggingface_hub[cli]" \
    torch>=2.0.0 \
    librosa \
    soundfile

# HeartLib klonlama (hata kontrolü ile)
RUN git clone https://github.com/HeartMuLa/heartlib.git && \
    cd heartlib && \
    pip install -e . && \
    echo "✅ HeartLib installed successfully"

# Model indirme (DETAYLI LOG ile)
RUN echo "📦 Starting model downloads..." && \
    mkdir -p ./ckpt && \
    echo "📥 Downloading HeartMuLaGen..." && \
    huggingface-cli download --local-dir './ckpt/HeartMuLaGen' 'HeartMuLa/HeartMuLaGen' && \
    echo "📥 Downloading HeartMuLa-oss-3B..." && \
    huggingface-cli download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year' && \
    echo "📥 Downloading HeartCodec-oss-20260123..." && \
    huggingface-cli download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss-20260123' && \
    echo "✅ All models downloaded successfully" && \
    ls -la ./ckpt/

# JSON dosyalarını ana dizine kopyala
RUN cp ./ckpt/HeartMuLaGen/*.json ./ckpt/ 2>/dev/null || echo "No JSON files to copy"

COPY handler.py .

# Container başlangıç komutu
CMD ["python", "-u", "handler.py"]
