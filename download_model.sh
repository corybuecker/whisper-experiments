#!/bin/bash

set -e
set -u
set -o pipefail

IFS=$'\n\t'

huggingface-cli download --local-dir /Volumes/AI/models/whisper openai/whisper-large-v3-turbo model.safetensors config.json tokenizer.json
huggingface-cli download --local-dir /Volumes/AI/models/crisper-whisper nyrahealth/CrisperWhisper model.safetensors config.json tokenizer.json
huggingface-cli download --local-dir /Volumes/AI/models/llama-3b-instruct meta-llama/Llama-3.2-3B-Instruct model-00001-of-00002.safetensors model-00002-of-00002.safetensors config.json tokenizer.json
huggingface-cli download --local-dir /Volumes/AI/models/llama-1b-instruct meta-llama/Llama-3.2-1B-Instruct model.safetensors config.json tokenizer.json
