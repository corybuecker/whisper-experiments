mod decoder;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self, audio};
use std::{error::Error, fs::File, path::PathBuf, str::FromStr};
use symphonia::{
    core::{
        audio::{AudioBufferRef, Signal},
        codecs::DecoderOptions,
        conv::FromSample,
        formats::FormatOptions,
        io::{MediaSourceStream, MediaSourceStreamOptions},
        meta::MetadataOptions,
        probe::Hint,
    },
    default::{get_codecs, get_probe},
};
use tokenizers::Tokenizer;
use tracing::{debug, Level};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .finish(),
    )?;

    let device = Device::new_metal(0)?;

    let tokenizer = Tokenizer::from_file(PathBuf::from("model/tokenizer.json")).unwrap();
    let config: whisper::Config =
        serde_json::from_slice(&std::fs::read(PathBuf::from("model/config.json"))?)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&["model/model.safetensors"], whisper::DTYPE, &device)?
    };

    let mut pcm_data: Vec<f32> = Vec::new();
    let sample = PathBuf::from_str("output.wav")?;
    let sample = File::open(sample)?;
    let mss = MediaSourceStream::new(Box::new(sample), MediaSourceStreamOptions::default());

    let metadata_opts = MetadataOptions::default();
    let format_opts = FormatOptions::default();
    let hint = Hint::default();

    let metadata = get_probe().format(&hint, mss, &format_opts, &metadata_opts)?;
    let mut format = metadata.format;
    let tracks = format.tracks();
    let track = &tracks[0];

    let decode_opts = DecoderOptions::default();
    let mut decoder = get_codecs().make(&track.codec_params, &decode_opts)?;

    while let Ok(packet) = format.next_packet() {
        let bytes = decoder.decode(&packet)?;

        match bytes {
            AudioBufferRef::F32(buf) => {
                pcm_data.extend(buf.chan(0));
            }
            AudioBufferRef::U8(data) => {
                pcm_data.extend(data.chan(0).iter().map(|b| f32::from_sample(*b)))
            }
            AudioBufferRef::U16(data) => {
                pcm_data.extend(data.chan(0).iter().map(|b| f32::from_sample(*b)))
            }
            AudioBufferRef::U24(data) => {
                pcm_data.extend(data.chan(0).iter().map(|b| f32::from_sample(*b)))
            }
            AudioBufferRef::U32(data) => {
                pcm_data.extend(data.chan(0).iter().map(|b| f32::from_sample(*b)))
            }
            AudioBufferRef::S8(data) => {
                pcm_data.extend(data.chan(0).iter().map(|b| f32::from_sample(*b)))
            }
            AudioBufferRef::S16(data) => {
                pcm_data.extend(data.chan(0).iter().map(|b| f32::from_sample(*b)))
            }
            AudioBufferRef::S24(data) => {
                pcm_data.extend(data.chan(0).iter().map(|b| f32::from_sample(*b)))
            }
            AudioBufferRef::S32(data) => {
                pcm_data.extend(data.chan(0).iter().map(|b| f32::from_sample(*b)))
            }
            AudioBufferRef::F64(data) => {
                pcm_data.extend(data.chan(0).iter().map(|b| f32::from_sample(*b)))
            }
        }
    }

    let mel_binary = include_bytes!("../melfilters128.bytes");
    let mut mel_filters = vec![0f32; mel_binary.len() / 4];
    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_binary, &mut mel_filters);

    let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
    let mel_length = mel.len();
    debug!("🚧 {:#?}", mel_length);

    let mel = Tensor::from_vec(mel, (1, 128, mel_length / 128), &device)?;

    let model = whisper::model::Whisper::load(&vb, config)?;
    let mut decoder = decoder::Decoder::new(model, tokenizer, &device)?;

    decoder.run(&mel)?;

    Ok(())
}
