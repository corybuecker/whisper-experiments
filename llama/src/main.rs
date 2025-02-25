use std::path::PathBuf;

use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::{self, Llama, LlamaConfig, LlamaEosToks},
};
use tokenizers::Tokenizer;
use tracing::{Level, debug};

#[tokio::main]
async fn main() -> Result<()> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .finish(),
    )?;

    let prompt = tokio::fs::read_to_string("./extract_ad_copy.prompt").await?;
    //let prompt = "What is a dog?".to_string();

    let device = Device::new_metal(0)?;

    let tokenizer = Tokenizer::from_file(PathBuf::from(
        "/Volumes/AI/models/llama-1b-instruct/tokenizer.json",
    ))
    .unwrap();
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(PathBuf::from(
        "/Volumes/AI/models/llama-1b-instruct/config.json",
    ))?)?;
    let weights = tokio::fs::read("/Volumes/AI/models/llama-1b-instruct/model.safetensors").await?;
    let vb = VarBuilder::from_slice_safetensors(&weights, DType::F16, &device)?;

    //  let tokenizer =
    //      Tokenizer::from_file(PathBuf::from("/Volumes/AI/models/llama-3b/tokenizer.json")).unwrap();
    //  let config: LlamaConfig = serde_json::from_slice(&std::fs::read(PathBuf::from(
    //      "/Volumes/AI/models/llama-3b/config.json",
    //  ))?)?;
    //  let filename = vec![
    //      "/Volumes/AI/models/llama-3b/model-00001-of-00002.safetensors".to_string(),
    //      "/Volumes/AI/models/llama-3b/model-00002-of-00002.safetensors".to_string(),
    //  ];
    //  let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filename, DType::F16, &device)? };

    let config = config.into_config(false);
    let eos_token_id = config
        .clone()
        .eos_token_id
        .ok_or(anyhow!("could not find EOT"))?;
    let model = Llama::load(vb, &config)?;
    let mut logits_processor = LogitsProcessor::from_sampling(8381264505028, Sampling::ArgMax);
    let mut cache = llama::Cache::new(true, DType::F16, &config, &device)?;

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|_| anyhow!("could not build tokenizer"))?
        .get_ids()
        .to_vec();

    debug!("🚧 {:#?}", tokens);
    let mut index_pos = 0_usize;
    for index in 0..1000 {
        let (context_size, _context_index) = if index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };

        let ctx = &tokens[tokens.len().saturating_sub(context_size)..];
        let tensor = Tensor::new(ctx, &device)?.unsqueeze(0)?;
        let logits = model.forward(&tensor, index_pos, &mut cache)?;
        let logits = logits.squeeze(0)?;

        index_pos += ctx.len();

        let token = logits_processor.sample(&logits)?;

        tokens.push(token);

        match eos_token_id {
            LlamaEosToks::Single(eos_tok_id) if token == eos_tok_id => {
                break;
            }
            LlamaEosToks::Multiple(ref eos_ids) if eos_ids.contains(&token) => {
                break;
            }
            _ => (),
        }
    }

    let text = tokenizer
        .decode(&tokens, false)
        .map_err(|_| anyhow!("could not parse tokens"))?;

    print!("{}", text);
    Ok(())
}
