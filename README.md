# Whisper Experiments

A Rust implementation for experimenting with the Whisper speech recognition model using the Candle framework.

## Dependencies

- Mac with GPU and Metal
- Huggingface CLI
- MP3 to WAV converter, e.g. FFMpeg

## Installation

1. Clone the repository:
```bash
git clone https://github.com/corybuecker/whisper-experiments.git
cd whisper-experiments
```

2. Install the Huggingface CLI if you haven't already.
```bash
brew install huggingface-cli
huggingface-cli login
```

3. Run the model download script:
```bash
./download_models.sh
```

## Usage

1. Convert your MP3 file to the required format:
   - 16-bit depth
   - 16kHz sample rate
   - WAV format
```bash
ffmpeg -i input.mp3 -ac 1 -acodec pcm_s16le -ar 16000 -map_metadata -1 output.wav
```

2. Run the project:
```bash
cargo run --release --features metal -- --file output.wav
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Candle framework and examples
- Huggingface

