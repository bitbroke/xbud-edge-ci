# xbud-edge-ci

This repository is a Linux ARM64 compatibility and safety-test harness.
It contains no production datasets, private recordings, credentials,
encryption keys, or proprietary model weights.

## Purpose
- Automated Linux ARM64 (aarch64) continuous integration verification on GitHub-hosted runners (`ubuntu-24.04-arm`).
- Fast contract testing for ASR worker facades, ALSA audio hardware wrappers, VAD audio processing, encrypted storage contracts, and scheduler crash recovery.
- Manifest and synthetic WAV hash integrity verification.
