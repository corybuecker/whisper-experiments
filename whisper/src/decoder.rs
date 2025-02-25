use anyhow::{Context, Result};
use candle_core::{D, Device, IndexOp, Tensor};
use candle_nn::ops::log_softmax;
use candle_transformers::models::whisper::{self as m, model::Whisper};
use tokenizers::Tokenizer;
use tracing::{debug, info};

pub struct Decoder {
    model: Whisper,
    tokenizer: Tokenizer,
    sot_token: u32,
    transcribe_token: u32,
    eot_token: u32,
    #[allow(dead_code)]
    no_speech_token: u32,
    start_timestamp_token: u32,
    mask_timestamps_tensor: Tensor,
    mask_non_timestamps_tensor: Tensor,
    mask_no_timestamps_tensor: Tensor,
    #[allow(dead_code)]
    mask_eot_tensor: Tensor,
    debug: bool,
}

fn build_mask_tensors(
    length: usize,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let mut mask_no_timestamps: Vec<f32> = vec![0.0; length];
    let mut mask_eot: Vec<f32> = vec![0.0; length];
    let mut mask_timestamps: Vec<f32> = vec![0.0; length];
    let mut mask_non_timestamps: Vec<f32> = vec![0.0; length];

    let no_timestamps_token: usize = tokenizer
        .token_to_id(m::NO_TIMESTAMPS_TOKEN)
        .context("could not get token")?
        .try_into()?;

    let start_timestamp_token: usize = tokenizer
        .token_to_id("<|0.00|>")
        .context("could not get token")?
        .try_into()?;

    let eot_token: usize = tokenizer
        .token_to_id(m::EOT_TOKEN)
        .context("could not get token")?
        .try_into()?;

    #[allow(clippy::needless_range_loop)]
    for i in start_timestamp_token..mask_timestamps.len() {
        mask_timestamps[i] = f32::NEG_INFINITY;
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..start_timestamp_token {
        mask_non_timestamps[i] = f32::NEG_INFINITY;
    }

    mask_no_timestamps[no_timestamps_token] = f32::NEG_INFINITY;
    mask_non_timestamps[eot_token] = 0.0;
    mask_eot[eot_token] = f32::NEG_INFINITY;

    Ok((
        Tensor::from_vec(mask_timestamps, length, device)?,
        Tensor::from_vec(mask_non_timestamps, length, device)?,
        Tensor::from_vec(mask_no_timestamps, length, device)?,
        Tensor::from_vec(mask_eot, length, device)?,
    ))
}

impl Decoder {
    pub fn new(model: Whisper, tokenizer: Tokenizer, device: &Device) -> Result<Self> {
        let sot_token = tokenizer
            .token_to_id(m::SOT_TOKEN)
            .context("could not get token")?;
        let transcribe_token = tokenizer
            .token_to_id(m::TRANSCRIBE_TOKEN)
            .context("could not get token")?;
        let start_timestamp_token = tokenizer
            .token_to_id("<|0.00|>")
            .context("could not get token")?;
        let eot_token = tokenizer
            .token_to_id(m::EOT_TOKEN)
            .context("could not get token")?;
        let no_speech_token = tokenizer
            .token_to_id(m::NO_SPEECH_TOKENS[1])
            .context("could not get token")?;

        let (
            mask_timestamps_tensor,
            mask_non_timestamps_tensor,
            mask_no_timestamps_tensor,
            mask_eot_tensor,
        ) = build_mask_tensors(model.config.vocab_size, &tokenizer, device)?;

        Ok(Self {
            model,
            tokenizer,
            sot_token,
            transcribe_token,
            eot_token,
            no_speech_token,
            mask_timestamps_tensor,
            mask_eot_tensor,
            mask_non_timestamps_tensor,
            start_timestamp_token,
            mask_no_timestamps_tensor,
            debug: false,
        })
    }

    fn suppress_smaller_timestamps(&mut self, timestamp: usize, device: &Device) -> Result<Tensor> {
        let mut mask: Vec<f32> = vec![0.0; self.model.config.vocab_size];
        let start_timestamp_token: usize = self
            .tokenizer
            .token_to_id("<|0.00|>")
            .context("could not get token")?
            .try_into()?;

        #[allow(clippy::needless_range_loop)]
        for i in start_timestamp_token..=(timestamp - 1) {
            mask[i] = f32::NEG_INFINITY;
        }

        let tensor = Tensor::from_vec(mask, self.model.config.vocab_size, device)?;

        Ok(tensor)
    }

    fn build_suppress_tokens(&mut self, tokens: &Tensor, device: &Device) -> Result<Tensor> {
        let mut zeroes = Tensor::zeros(self.model.config.vocab_size, m::DTYPE, device)?;
        zeroes = zeroes.add(&self.mask_no_timestamps_tensor)?;

        if tokens.dims().len() != 1 {
            debug!("incorrect tensor dims");
            return Ok(zeroes);
        }

        let mut tokens: Vec<u32> = tokens.to_vec1()?;

        if let (Some(last), Some(second_last)) = (tokens.pop(), tokens.pop()) {
            if last > self.start_timestamp_token {
                if second_last > self.start_timestamp_token {
                    // both are timestmps, no more timestamps; cannot be EOT since there were
                    // two timestamps
                    zeroes = zeroes.add(&self.mask_timestamps_tensor)?;
                } else if second_last != 50360 {
                    //only one timestamp, looking for another, ge timestamp or EOT
                    zeroes = zeroes.add(&self.mask_non_timestamps_tensor)?;
                    //zeroes = zeroes.add(&self.mask_eot_tensor)?;
                    let smaller_timestamps =
                        self.suppress_smaller_timestamps(last.try_into().unwrap(), device)?;
                    zeroes = zeroes.add(&smaller_timestamps)?;
                }
            }
        }

        Ok(zeroes)
    }
    pub fn decode(&mut self, mel: &Tensor) -> Result<Vec<u32>> {
        let audio_features = self.model.encoder.forward(mel, true)?;
        let en_token = self.tokenizer.token_to_id("<|en|>").unwrap();
        let mut tokens = vec![self.sot_token];
        let sample_len = self.model.config.max_target_positions / 2;

        tokens.push(en_token);
        tokens.push(self.transcribe_token);
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            let suppress_tokens = self.build_suppress_tokens(&tokens_t, mel.device())?;

            let tokens_t = tokens_t.unsqueeze(0)?;

            let ys = self
                .model
                .decoder
                // Curious...what does the cache do?
                .forward(&tokens_t, &audio_features, i == 0)?;

            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let logits = logits.add(&suppress_tokens)?;

            let log_sm_logits = log_softmax(&logits, D::Minus1)?;
            let non_timestamp_prob_tensor =
                log_sm_logits.i(..self.start_timestamp_token as usize)?;
            let largest_non_timestamp_prob =
                non_timestamp_prob_tensor.max(D::Minus1)?.to_scalar()?;

            let timestamp_prob_tensor = log_sm_logits
                .i(self.start_timestamp_token as usize..)?
                .log_sum_exp(D::Minus1)?;
            let timestamp_prob = timestamp_prob_tensor.to_scalar::<f32>()?;

            let logits = if timestamp_prob > largest_non_timestamp_prob {
                logits.add(&self.mask_non_timestamps_tensor)?
            } else {
                logits
            };

            let next_token = {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
                // let prs = softmax(&(&logits / 0.2)?, 0)?;
                // let logits_v: Vec<f32> = prs.to_vec1()?;
                // let distr = WeightedIndex::new(&logits_v)?;
                // distr.sample(&mut rng) as u32
            };

            tokens.push(next_token);

            if next_token == self.eot_token || tokens.len() > self.model.config.max_target_positions
            {
                break;
            }
        }

        Ok(tokens)
    }

    pub fn run(&mut self, mel: &Tensor) -> Result<Vec<String>> {
        let mut segments = vec![];
        debug!("starting model run");
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let input_stride = m::N_FRAMES / self.model.config.max_source_positions;

        while seek < content_frames {
            self.debug = false;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let tokens = self.decode(&mel_segment)?;

            let all_tokens = tokens.clone();
            let [_second_last, last, _] = all_tokens[all_tokens.len() - 3..] else {
                panic!()
            };

            if last > self.start_timestamp_token {
                let diff = (last - self.start_timestamp_token) as usize;
                seek += diff * input_stride;
            } else {
                seek += segment_size;
            }
            let decode_all = self.tokenizer.decode(&all_tokens, false).unwrap();
            info!("🚧 {:#?}", &decode_all);
            segments.push(decode_all);
        }

        Ok(segments)
    }
}
