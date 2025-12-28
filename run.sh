#!/bin/bash
# Suppress ALSA audio errors in WSL by redirecting stderr
# The audio warnings won't prevent the program from running in dev mode
echo "Running in development mode (audio warnings are harmless)..."
cargo run "$@" 2>&1 | grep -v "ALSA\|PCM\|snd_"
