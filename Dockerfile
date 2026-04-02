FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /workspace

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları - AYRI AYRI VE DETAYLI LOG İLE
RUN pip install --no-cache-dir --upgrade pip

# Runpod'u ÖNCE kur (en kritik paket)
RUN pip install --no-cache-dir runpod && \
    python -c "import runpod; print(f'✅ Runpod version: {runpod.__version__}')"

# Diğer paketler
RUN pip install --no-cache-dir \
    huggingface_hub \
    torch>=2.0.0 \
    librosa \
    soundfile

# HeartLib klonlama
RUN git clone https://github.com/HeartMuLa/heartlib.git && \
    cd heartlib && \
    pip install -e . && \
    echo "✅ HeartLib installed"

# Model indirme
RUN mkdir -p /workspace/ckpt && \
    cd /workspace && \
    echo "📥 Downloading models..." && \
    huggingface-cli download --local-dir ./ckpt/HeartMuLaGen HeartMuLa/HeartMuLaGen && \
    huggingface-cli download --local-dir ./ckpt/HeartMuLa-oss-3B HeartMuLa/HeartMuLa-oss-3B-happy-new-year && \
    huggingface-cli download --local-dir ./ckpt/HeartCodec-oss HeartMuLa/HeartCodec-oss-20260123 && \
    cp ./ckpt/HeartMuLaGen/*.json ./ckpt/ 2>/dev/null || true && \
    echo "✅ Models downloaded"

# Test import (build sırasında kontrol et)
RUN python -c "import runpod; import torch; from heartlib.pipelines.music_generation import MusicGenerationPipeline; print('✅ All imports OK')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
