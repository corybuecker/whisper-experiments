use anyhow::{anyhow, Result};
use regex::Regex;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Segment {
    start_time: f32,
    end_time: f32,
    text: String,
}

#[derive(Debug, Serialize)]
pub struct Transcript {
    #[serde(skip_serializing)]
    raw_segments: Vec<String>,
    segments: Vec<Segment>,
    text: String,
}

impl Transcript {
    pub fn new(segments: Vec<String>) -> Result<Self> {
        let raw_segments = segments;

        let mut transcript = Transcript {
            raw_segments,
            segments: vec![],
            text: String::new(),
        };

        transcript.parse_segments()?;

        transcript.text = transcript
            .segments
            .iter()
            .map(|s| s.text.clone())
            .collect::<Vec<String>>()
            .join(" ");

        Ok(transcript)
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    fn parse_segments(&mut self) -> Result<()> {
        let mut current_offset = 0.0;
        let regex = Regex::new(r"<\|(?<time>\d{1,2}\.\d{2})\|>")?;

        for raw_segment in &self.raw_segments {
            let it = regex.captures_iter(raw_segment);
            let mut segment_timestamps = vec![];
            for part in it {
                let (_, [time]) = part.extract();
                segment_timestamps.push(time);
            }

            let segment_timestamps: Vec<(String, String)> = segment_timestamps
                .iter()
                .zip(segment_timestamps.iter().skip(1))
                .map(|(a, b)| (String::from(*a), String::from(*b)))
                .collect();
            for (start, end) in segment_timestamps {
                let start_index = raw_segment
                    .find(&start)
                    .ok_or(anyhow!("could not get index"))?;

                let end_index = raw_segment
                    .find(&end)
                    .ok_or(anyhow!("could not get index"))?;

                let substring =
                    String::from(&raw_segment[start_index - 2..end_index + end.len() + 2]);

                let substring = regex
                    .replace_all(&substring, "")
                    .into_owned()
                    .trim()
                    .to_owned();

                if !substring.is_empty() {
                    let start = start.parse::<f32>()?;
                    let end = end.parse::<f32>()?;
                    self.segments.push(Segment {
                        start_time: start + current_offset,
                        end_time: end + current_offset,
                        text: substring,
                    });
                }
            }

            current_offset = self.segments.last().ok_or(anyhow!("no segments"))?.end_time;
        }

        Ok(())
    }
}
