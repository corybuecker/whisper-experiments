mod file;

use anyhow::Result;
use byteorder::{ByteOrder, LittleEndian};
use candle_core::{Device, Tensor};
use candle_transformers::models::whisper::{audio::pcm_to_mel, Config};
use hound::WavReader;
use std::{fmt, fs::read};
use tracing::debug;

const SAMPLE: f32 = 32_768.0;

#[derive(Debug)]
pub struct SignalProcessingError {}
impl std::fmt::Display for SignalProcessingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SuperError is here!")
    }
}
impl std::error::Error for SignalProcessingError {}

pub fn load_audio_file(filename: &str, config: &Config, device: &Device) -> Result<Tensor> {
    let mut wav = WavReader::open(filename).or(Err(SignalProcessingError {}))?;
    //   let mut wav = WavReader::open("full.wav")?;

    let spec = wav.spec();

    debug!(
        "🚧 Audio specs - Channels: {}, Sample rate: {}",
        spec.channels, spec.sample_rate
    );

    let pcm_data: Vec<f32> = wav
        .samples::<i32>()
        .filter_map(|sample| sample.ok())
        .map(|sample| sample as f32 / SAMPLE)
        .collect();

    debug!("🚧 PCM data {:#?}", pcm_data.len());

    let mel_binary_128 = read("melfilters128.bytes")?;
    let byte_length = 128;
    let mut mel_filters = vec![0f32; mel_binary_128.len() / 4];

    LittleEndian::read_f32_into(&mel_binary_128, &mut mel_filters);

    let mel = pcm_to_mel(config, &pcm_data, &mel_filters);
    let mel_length = mel.len();

    Ok(Tensor::from_vec(
        mel,
        (1, byte_length, mel_length / byte_length),
        device,
    )?)
}
