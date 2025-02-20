mod decoder;

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self, audio};
use hound::WavReader;
use std::{fs::read, path::PathBuf};
use tokenizers::Tokenizer;
use tracing::{debug, Level};

#[tokio::main]
async fn main() -> Result<()> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .finish(),
    )?;

    let device = Device::new_metal(0)?;

    let tokenizer =
        Tokenizer::from_file(PathBuf::from("/Volumes/AI/models/whisper/tokenizer.json")).unwrap();
    let config: whisper::Config = serde_json::from_slice(&std::fs::read(PathBuf::from(
        "/Volumes/AI/models/whisper/config.json",
    ))?)?;

    let mut wav = WavReader::open("output.mono.wav")?;
    //   let mut wav = WavReader::open("full.wav")?;

    let spec = wav.spec();

    debug!(
        "🚧 Audio specs - Channels: {}, Sample rate: {}",
        spec.channels, spec.sample_rate
    );

    let pcm_data: Vec<f32> = wav
        .samples::<i32>()
        .filter_map(|sample| sample.ok())
        .map(|sample| sample as f32 / 32_768.0)
        .collect();

    debug!("🚧 PCM data {:#?}", pcm_data.len());

    let mel_binary_128 = read("melfilters128.bytes")?;
    let byte_length = 128;
    let mut mel_filters = vec![0f32; mel_binary_128.len() / 4];

    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
        &mel_binary_128,
        &mut mel_filters,
    );

    let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
    let mel_length = mel.len();
    let mel = Tensor::from_vec(mel, (1, byte_length, mel_length / byte_length), &device)?;

    let file = read("/Volumes/AI/models/whisper/model.safetensors")?;
    let vb = VarBuilder::from_slice_safetensors(&file, whisper::DTYPE, &device)?;
    let model = whisper::model::Whisper::load(&vb, config)?;
    let mut decoder = decoder::Decoder::new(model, tokenizer, &device)?;

    decoder.run(&mel)?;

    Ok(())
}
