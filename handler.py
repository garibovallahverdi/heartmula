import runpod
import torch
import sys
import traceback
import tempfile
import base64
import os

# Global exception handler
def global_exception_handler(exc_type, exc_value, exc_traceback):
    print("=" * 50)
    print("UNHANDLED EXCEPTION")
    print("=" * 50)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)

sys.excepthook = global_exception_handler

print("=" * 50)
print("Starting HeartMuLa Worker")
print("=" * 50)

# CUDA kontrolü
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Model yükleme
print("\n📦 Loading HeartMuLa pipeline...")
try:
    # Önce klasör yapısını kontrol et
    import os
    print("Checking checkpoint directory:")
    if os.path.exists("./ckpt"):
        for item in os.listdir("./ckpt"):
            print(f"  - {item}")
    else:
        print("  ❌ ./ckpt directory not found!")
    
    pipe = MusicGenerationPipeline(
        model_path="./ckpt",
        version="3B",
        lazy_load=True,
        mula_device="cuda",
        codec_device="cuda"
    )
    print("✅ Pipeline loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load pipeline: {e}")
    traceback.print_exc()
    sys.exit(1)

def handler(job):
    """RunPod endpoint handler"""
    print(f"\n📨 Received job: {job.get('id', 'unknown')}")
    
    try:
        job_input = job.get("input", {})
        
        # Parametreleri al
        lyrics = job_input.get("lyrics", "")
        tags = job_input.get("tags", "pop,happy,melodic")
        max_audio_length_ms = job_input.get("max_audio_length_ms", 30000)
        topk = job_input.get("topk", 50)
        temperature = job_input.get("temperature", 1.0)
        cfg_scale = job_input.get("cfg_scale", 1.5)
        
        print(f"  🎵 Tags: {tags}")
        print(f"  ⏱️  Duration: {max_audio_length_ms/1000}s")
        print(f"  🎨 Temperature: {temperature}")
        
        # Lyrics'i geçici dosyaya yaz
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(lyrics)
            lyrics_path = f.name
        print(f"  📝 Lyrics saved to: {lyrics_path}")
        
        # Müzik üret
        output_path = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
        print(f"  🎶 Generating music...")
        
        pipe.generate(
            lyrics_path=lyrics_path,
            tags=tags,
            save_path=output_path,
            max_audio_length_ms=max_audio_length_ms,
            topk=topk,
            temperature=temperature,
            cfg_scale=cfg_scale
        )
        
        # Base64'e çevir
        with open(output_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Temizlik
        os.unlink(lyrics_path)
        os.unlink(output_path)
        
        print(f"  ✅ Music generated successfully!")
        
        return {
            "status": "COMPLETED",
            "audio_base64": audio_b64,
            "duration_ms": max_audio_length_ms
        }
        
    except Exception as e:
        print(f"❌ Error in handler: {e}")
        traceback.print_exc()
        return {
            "status": "FAILED",
            "error": str(e)
        }

# RunPod serverless başlat
print("\n🚀 Starting RunPod serverless...")
runpod.serverless.start({"handler": handler})
