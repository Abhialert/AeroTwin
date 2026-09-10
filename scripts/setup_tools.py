"""
AeroTwin - Setup Script: Download and configure required tools
==============================================================
Downloads COLMAP and FFmpeg pre-built binaries for Windows.

Usage:
    python scripts/setup_tools.py

This will create:
    tools/
    +-- colmap/     - COLMAP binaries
    +-- ffmpeg/     - FFmpeg binaries
"""

import os
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import zipfile
import shutil
import urllib.request
from pathlib import Path


TOOLS_DIR = Path("tools")

# COLMAP pre-built Windows binary (CUDA)
# Using the latest stable release from GitHub (v4.2.0)
COLMAP_URL = "https://github.com/colmap/colmap/releases/download/4.2.0/colmap-x64-windows-cuda.zip"
COLMAP_ZIP = "colmap.zip"

# FFmpeg static build for Windows
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_ZIP = "ffmpeg.zip"


def download_file(url, dest_path, description=""):
    """Download a file with progress indication."""
    print(f"\n  Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=progress_hook)
        print(f"\n  ✓ Download complete: {dest_path}")
        return True
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        print(f"  Please download manually from: {url}")
        return False


def setup_colmap():
    """Download and extract COLMAP."""
    colmap_dir = TOOLS_DIR / "colmap"
    
    # Check if already installed
    colmap_exe = None
    for candidate in [colmap_dir / "COLMAP.bat", colmap_dir / "colmap.exe"]:
        if candidate.exists():
            colmap_exe = candidate
            break
    
    if colmap_exe:
        print(f"\n  ✓ COLMAP already installed: {colmap_exe}")
        return str(colmap_exe)
    
    # Also check if in system PATH
    if shutil.which("colmap"):
        print(f"\n  ✓ COLMAP found in system PATH: {shutil.which('colmap')}")
        return shutil.which("colmap")
    
    # Download
    os.makedirs(TOOLS_DIR, exist_ok=True)
    zip_path = TOOLS_DIR / COLMAP_ZIP
    
    if not download_file(COLMAP_URL, zip_path, "COLMAP"):
        return None
    
    # Extract
    print(f"\n  Extracting COLMAP...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(TOOLS_DIR))
        
        # Find the extracted directory and rename to 'colmap'
        extracted_dirs = [d for d in TOOLS_DIR.iterdir() if d.is_dir() and "COLMAP" in d.name.upper()]
        if extracted_dirs:
            extracted = extracted_dirs[0]
            if extracted.name != "colmap":
                target = TOOLS_DIR / "colmap"
                if target.exists():
                    shutil.rmtree(str(target))
                extracted.rename(target)
        
        # Clean up zip
        zip_path.unlink()
        
        # Find executable
        for candidate in [
            TOOLS_DIR / "colmap" / "COLMAP.bat",
            TOOLS_DIR / "colmap" / "colmap.exe",
            TOOLS_DIR / "colmap" / "bin" / "colmap.exe",
        ]:
            if candidate.exists():
                print(f"  ✓ COLMAP installed: {candidate}")
                return str(candidate)
        
        # List what we got
        print(f"  Contents of tools/colmap/:")
        for item in (TOOLS_DIR / "colmap").rglob("*"):
            if item.is_file() and item.suffix in [".exe", ".bat"]:
                print(f"    {item}")
        
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return None
    
    return None


def setup_ffmpeg():
    """Download and extract FFmpeg."""
    ffmpeg_dir = TOOLS_DIR / "ffmpeg"
    
    # Check if already installed
    ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
    if ffmpeg_exe.exists():
        print(f"\n  ✓ FFmpeg already installed: {ffmpeg_exe}")
        return str(ffmpeg_exe)
    
    # Check system PATH
    if shutil.which("ffmpeg"):
        print(f"\n  ✓ FFmpeg found in system PATH: {shutil.which('ffmpeg')}")
        return shutil.which("ffmpeg")
    
    # Download
    os.makedirs(TOOLS_DIR, exist_ok=True)
    zip_path = TOOLS_DIR / FFMPEG_ZIP
    
    if not download_file(FFMPEG_URL, zip_path, "FFmpeg"):
        return None
    
    # Extract
    print(f"\n  Extracting FFmpeg...")
    try:
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(TOOLS_DIR))
        
        # Find the extracted directory
        extracted_dirs = [d for d in TOOLS_DIR.iterdir() if d.is_dir() and "ffmpeg" in d.name.lower()]
        if extracted_dirs:
            extracted = extracted_dirs[0]
            # FFmpeg extracts to a versioned directory like ffmpeg-7.0-essentials_build
            # Move bin contents to tools/ffmpeg/
            bin_dir = extracted / "bin"
            os.makedirs(ffmpeg_dir, exist_ok=True)
            
            if bin_dir.exists():
                for f in bin_dir.iterdir():
                    shutil.copy2(str(f), str(ffmpeg_dir / f.name))
                # Clean up extracted directory
                shutil.rmtree(str(extracted))
        
        # Clean up zip
        zip_path.unlink()
        
        if ffmpeg_exe.exists():
            print(f"  ✓ FFmpeg installed: {ffmpeg_exe}")
            return str(ffmpeg_exe)
    
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return None
    
    return None


def main():
    print(f"{'='*60}")
    print(f"  AeroTwin — Tool Setup")
    print(f"{'='*60}")
    
    results = {}
    
    # Setup COLMAP
    print(f"\n{'─'*60}")
    print(f"  COLMAP (3D Reconstruction Engine)")
    print(f"{'─'*60}")
    colmap_path = setup_colmap()
    results["colmap"] = colmap_path
    
    # Setup FFmpeg
    print(f"\n{'─'*60}")
    print(f"  FFmpeg (Video Processing)")
    print(f"{'─'*60}")
    ffmpeg_path = setup_ffmpeg()
    results["ffmpeg"] = ffmpeg_path
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  SETUP SUMMARY")
    print(f"{'='*60}")
    
    all_ok = True
    for tool, path in results.items():
        status = "✓" if path else "❌"
        print(f"  {status} {tool:15s} → {path or 'NOT INSTALLED'}")
        if not path:
            all_ok = False
    
    if all_ok:
        print(f"\n  All tools ready! You can now run the pipeline.")
    else:
        print(f"\n  ⚠️  Some tools are missing. See errors above.")
        print(f"  You can install them manually and re-run this script.")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
