"""
AeroTwin - Sample Data Downloader
Downloads a real sample drone flight video for testing the pipeline.
"""
import os
import sys
import urllib.request
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SAMPLE_URL = "https://user-images.githubusercontent.com/870796/189461690-122f4e64-a66e-40f0-ac4b-68258a8abe7e.mov"
OUTPUT_PATH = Path("data/input/drone_flight_sample.mov")

def download_sample():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists() and OUTPUT_PATH.stat().st_size > 1000000:
        print(f"Sample video already present: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / (1024*1024):.2f} MB)")
        return str(OUTPUT_PATH)
    
    print(f"Downloading sample drone flight video...")
    print(f"URL: {SAMPLE_URL}")
    print(f"Target: {OUTPUT_PATH}")
    
    def hook(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 / total_size
            mb = (count * block_size) / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            print(f"\rProgress: {pct:.1f}% ({mb:.1f}/{tot_mb:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(SAMPLE_URL, str(OUTPUT_PATH), reporthook=hook)
    print(f"\nDownload complete! Size: {OUTPUT_PATH.stat().st_size / (1024*1024):.2f} MB")
    return str(OUTPUT_PATH)

if __name__ == "__main__":
    download_sample()
