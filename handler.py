import runpod
import torch
from heartlib.pipelines.music_generation import MusicGenerationPipeline

# Globalde model yükleme (her worker başladığında 1 kere çalışır)
print("Loading HeartMuLa model...")
pipe = MusicGenerationPipeline(model_path="./ckpt", version="3B")
pipe.to("cuda")
print("Model loaded.")

def handler(job):
    """RunPod'un çağıracağı ana fonksiyon"""
    job_input = job["input"]
    
    lyrics = job_input.get("lyrics", "")
    tags = job_input.get("tags", "pop, happy")
    duration_seconds = job_input.get("duration_seconds", 30)
    temperature = job_input.get("temperature", 1.0)
    cfg_scale = job_input.get("cfg_scale", 1.5)
    topk = job_input.get("topk", 50)
    
    # Müzik üret
    audio_array = pipe.generate(
        lyrics=lyrics,
        tags=tags,
        duration_seconds=duration_seconds,
        temperature=temperature,
        cfg_scale=cfg_scale,
        topk=topk
    )
    
    # Audio'yu base64 veya dosya olarak döndür
    # (tercihine göre)
    return {
        "status": "COMPLETED",
        "audio_base64": audio_array_to_base64(audio_array),
        "duration": duration_seconds
    }

# RunPod serverless başlat
runpod.serverless.start({"handler": handler})
