mod decoder;
mod signal_processing;

use anyhow::Result;
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self};
use signal_processing::load_audio_file;
use std::{env::args, fs::read, path::PathBuf};
use tokenizers::Tokenizer;
use tracing::{debug, Level};

#[tokio::main]
async fn main() -> Result<()> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .finish(),
    )?;

    let args: Vec<String> = args().collect();
    debug!("🚧 {:#?}", args);
    let file_arg = args[2].clone();

    let device = Device::new_metal(0)?;
    let tokenizer =
        Tokenizer::from_file(PathBuf::from("/Volumes/AI/models/whisper/tokenizer.json")).unwrap();
    let config: whisper::Config = serde_json::from_slice(&std::fs::read(PathBuf::from(
        "/Volumes/AI/models/whisper/config.json",
    ))?)?;

    let mel = load_audio_file(&file_arg, &config, &device)?;

    let file = read("/Volumes/AI/models/whisper/model.safetensors")?;
    let vb = VarBuilder::from_slice_safetensors(&file, whisper::DTYPE, &device)?;
    let model = whisper::model::Whisper::load(&vb, config)?;
    let mut decoder = decoder::Decoder::new(model, tokenizer, &device)?;

    decoder.run(&mel)?;

    Ok(())
}
