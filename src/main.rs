mod decoder;
mod signal_processing;

use anyhow::Result;
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self};
use signal_processing::load_audio_file;
use std::{env::args, fs::read, path::PathBuf};
use tokenizers::Tokenizer;
use tracing::Level;

#[tokio::main]
async fn main() -> Result<()> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .finish(),
    )?;

    let file = args()
        .position(|arg| arg == "--file")
        .and_then(|index| args().nth(index + 1))
        .ok_or_else(|| anyhow::anyhow!("Missing required --file argument"))?;

    let device = Device::new_metal(0)?;
    let tokenizer =
        Tokenizer::from_file(PathBuf::from("/Volumes/AI/models/whisper/tokenizer.json")).unwrap();
    let config: whisper::Config = serde_json::from_slice(&std::fs::read(PathBuf::from(
        "/Volumes/AI/models/whisper/config.json",
    ))?)?;

    let mel = load_audio_file(&file, &config, &device)?;

    let file = read("/Volumes/AI/models/whisper/model.safetensors")?;
    let vb = VarBuilder::from_slice_safetensors(&file, whisper::DTYPE, &device)?;
    let model = whisper::model::Whisper::load(&vb, config)?;
    let mut decoder = decoder::Decoder::new(model, tokenizer, &device)?;

    decoder.run(&mel)?;

    Ok(())
}
