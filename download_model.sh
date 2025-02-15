#!/bin/bash

set -e
set -u
set -o pipefail

IFS=$'\n\t'

mkdir -p /Volumes/AI/models/whisper

# huggingface-cli download --local-dir /Volumes/AI/models/whisper openai/whisper-small model.safetensors config.json tokenizer.json
huggingface-cli download --local-dir /Volumes/AI/models/whisper openai/whisper-large-v3-turbo model.safetensors config.json tokenizer.json
