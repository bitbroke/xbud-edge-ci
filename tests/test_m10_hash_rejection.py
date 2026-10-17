import os
import sys
import shutil
import subprocess
import pytest

def test_hash_rejection(tmp_path):
    # We will use the test_fixture dataset
    original_dataset = "datasets/test_fixture"
    dataset = str(tmp_path / "test_fixture")
    
    # Copy the dataset to tmp_path to avoid mutating the repo
    shutil.copytree(original_dataset, dataset)
    
    manifest_path = os.path.join(dataset, "manifest.json")
    wav_path = os.path.join(dataset, "dummy.wav")
    
    cmd = [
        sys.executable, "benchmarks/run_m8.py",
        "--dataset", dataset,
        "--verify-only"
    ]
    
    # 1. First, make sure the hash matches (which it should right now)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
    assert result.returncode == 0, f"Expected success but got failure: {result.stderr} {result.stdout}"
    assert "Verify-only mode: manifest and hashes validated successfully." in result.stdout
    assert "TRACK A: ASR ONLY" not in result.stdout, "Whisper should not be launched in verify-only mode"
    
    # 2. Modify one byte of the dummy.wav
    with open(wav_path, "r+b") as f:
        f.seek(100) # Arbitrary byte
        byte = f.read(1)
        # Flip a bit
        f.seek(100)
        f.write(bytes([byte[0] ^ 1]))
        
    # 3. Run the benchmark again, it should fail
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
    assert result.returncode == 1, "Expected benchmark to fail due to hash mismatch"
    assert "ERROR: Hash mismatch" in result.stdout
