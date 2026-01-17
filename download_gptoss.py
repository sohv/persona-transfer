#!/usr/bin/env python3
"""
Download GPT-OSS 20B model in GGUF format.
This script downloads the quantized GGUF model file for use with llama.cpp.
"""
import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Model details
MODEL_NAME = "openai_gpt-oss-20b-Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/gpt-oss-20b-GGUF/resolve/main/openai_gpt-oss-20b-Q4_K_M.gguf"
MODEL_SIZE_GB = 12.0

# Download directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def download_with_progress(url, destination):
    """Download file with progress bar."""
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=destination.name) as t:
        urllib.request.urlretrieve(url, filename=destination, reporthook=t.update_to)

def main():
    model_path = MODELS_DIR / MODEL_NAME
    
    print("GPT-OSS 20B Model Downloader")
    print("=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"Size: ~{MODEL_SIZE_GB}GB")
    print(f"Destination: {model_path}")
    print("=" * 50)
    print()
    
    # Check if already downloaded
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        response = input("Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return
        print("Removing existing file...")
        model_path.unlink()
    
    # Check disk space
    stat = os.statvfs(MODELS_DIR)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    if free_gb < MODEL_SIZE_GB + 1:
        print(f"WARNING: Low disk space. Free: {free_gb:.1f}GB, Required: ~{MODEL_SIZE_GB}GB")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Download cancelled.")
            return
    
    print(f"Downloading from: {MODEL_URL}")
    print(f"This will take a while depending on your internet connection...")
    print()
    
    try:
        download_with_progress(MODEL_URL, model_path)
        print()
        print("=" * 50)
        print(f"Download complete! Model saved to: {model_path}")
        print("=" * 50)
        print()
        print("You can now use this model by selecting 'gpt-oss-20b' in the web interface.")
        
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        if model_path.exists():
            print("Removing incomplete file...")
            model_path.unlink()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError downloading model: {e}")
        if model_path.exists():
            print("Removing incomplete file...")
            model_path.unlink()
        sys.exit(1)

if __name__ == "__main__":
    main()
