import runpod
import torch
from heartlib.pipelines.music_generation import MusicGenerationPipeline

# Globalde pipeline'ı başlat (her worker başladığında 1 kere)
print("Loading HeartMuLa pipeline...")
pipe = MusicGenerationPipeline(
    model_path="./ckpt",
    version="3B",
    lazy_load=True,  # GPU belleğini korumak için önemli
    mula_device="cuda",
    codec_device="cuda"
)
print("Pipeline loaded successfully.")

def handler(job):
    """RunPod endpoint handler"""
    job_input = job["input"]
    
    # Kullanıcıdan gelen parametreler
    lyrics = job_input.get("lyrics", "")
    tags = job_input.get("tags", "pop,happy,melodic")
    max_audio_length_ms = job_input.get("max_audio_length_ms", 30000)
    topk = job_input.get("topk", 50)
    temperature = job_input.get("temperature", 1.0)
    cfg_scale = job_input.get("cfg_scale", 1.5)
    
    # Eğer lyrics dosya yolu değil de direkt metin olarak geldiyse,
    # geçici bir dosyaya yaz
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(lyrics)
        lyrics_path = f.name
    
    # Müzik üret
    output_path = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
    
    pipe.generate(
        lyrics_path=lyrics_path,
        tags=tags,
        save_path=output_path,
        max_audio_length_ms=max_audio_length_ms,
        topk=topk,
        temperature=temperature,
        cfg_scale=cfg_scale
    )
    
    # Base64'e çevir ve döndür
    import base64
    with open(output_path, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    return {
        "status": "COMPLETED",
        "audio_base64": audio_b64,
        "duration_ms": max_audio_length_ms
    }

# RunPod serverless başlat
runpod.serverless.start({"handler": handler})
