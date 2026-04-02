FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Çalışma dizini
WORKDIR /workspace

# Sistem bağımlılıkları (gerekirse)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıkları
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# HeartMuLa kodunu kopyala (veya direkt pip ile kur)
# Eğer heartlib pip'te yoksa repoyu klonlayıp kurman gerek:
RUN git clone https://github.com/HeartMuLa/heartlib.git && \
    cd heartlib && \
    pip install -e .

# Modelleri indir (build sırasında veya runtime'da)
# Build'de indirmek imaj boyutunu büyütür ama cold start'ı hızlandırır
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='HeartMuLa/HeartMuLa-oss-3B-happy-new-year', local_dir='./ckpt/HeartMuLa-oss-3B'); \
snapshot_download(repo_id='HeartMuLa/HeartCodec-oss', local_dir='./ckpt/HeartCodec-oss'); \
"

# handler.py'yi kopyala
COPY handler.py .

# RunPod'ın beklendiği gibi çalıştır
CMD ["python", "-u", "handler.py"]
