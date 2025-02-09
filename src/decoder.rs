use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::whisper::{self as m, model::Whisper};
use tokenizers::Tokenizer;
use tracing::debug;

pub struct Decoder {
    model: Whisper,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
}

impl Decoder {
    pub fn new(model: Whisper, tokenizer: Tokenizer, device: &Device) -> Result<Self> {
        let no_timestamps_token = tokenizer.token_to_id(m::NO_TIMESTAMPS_TOKEN).unwrap();
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i) || i == no_timestamps_token {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = tokenizer.token_to_id(m::SOT_TOKEN).unwrap();
        let transcribe_token = tokenizer.token_to_id(m::TRANSCRIBE_TOKEN).unwrap();
        let eot_token = tokenizer.token_to_id(m::EOT_TOKEN).unwrap();
        Ok(Self {
            model,
            tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            eot_token,
        })
    }

    pub fn decode(&mut self, mel: &Tensor) -> Result<Vec<u32>> {
        let model = &mut self.model;
        let audio_features = model.encoder.forward(mel, true)?;
        let sample_len = model.config.max_target_positions / 2;
        let mut tokens = vec![self.sot_token];
        let en_token = self.tokenizer.token_to_id("<|en|>").unwrap();

        tokens.push(en_token);
        tokens.push(self.transcribe_token);

        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder.forward(&tokens_t, &audio_features, i == 0)?;

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            if next_token == self.eot_token || tokens.len() > model.config.max_target_positions {
                break;
            }
        }

        Ok(tokens)
    }

    pub fn run(&mut self, mel: &Tensor) -> Result<Vec<u32>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        while seek < content_frames {
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let tokens = self.decode(&mel_segment)?;

            let all_tokens = tokens.clone();

            let decode_all = self.tokenizer.decode(&all_tokens, false).unwrap();
            debug!("🚧 {:#?}", decode_all);

            seek += segment_size;
        }
        Ok(vec![])
    }
}
