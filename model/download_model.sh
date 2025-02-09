#!/bin/bash

set -e
set -u
set -o pipefail

IFS=$'\n\t'

rm -f model.safetensors config.json tokenizer.json
huggingface-cli download --local-dir . openai/whisper-large-v3-turbo model.safetensors config.json tokenizer.json
