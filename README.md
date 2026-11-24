# audio-qc-manifest-auditor

A zero-dependency Python 3.10+ CLI tool for **audio captioning, transcription, and multimodal manifest quality control (QC)**.

Designed for audio annotation, captioning, and speech/sound dataset pipelines to catch invalid or overlapping timestamps, duration outliers, empty or placeholder transcripts, and duplicate segment IDs prior to dataset delivery.

## QC Checks

- **Schema & Timestamp Bounds (`AUD-001`–`AUD-004`, `TIM-001`):** Validates required fields (`clip_id`, `start_sec`, `end_sec`) and rejects negative or non-increasing timestamp intervals.
- **Duration Outliers (`TIM-002`–`TIM-003`):** Flags segments shorter than `--min-duration` or longer than `--max-duration`.
- **Segment Overlap Detection (`TIM-004`):** Sorts segments per `clip_id` and flags overlapping time ranges.
- **Transcript / Caption Quality (`TXT-001`–`TXT-002`):** Flags empty transcripts as `ERROR` and short or placeholder strings (`n/a`, `tbd`, `[inaudible]`) as `WARN`.
- **Duplicate Detection (`DUP-SEG`):** Detects duplicate `(clip_id, segment_id)` pairs.

## Quickstart

Requires **Python 3.10+** (standard library only).

```bash
git clone https://github.com/B5IVE/audio-qc-manifest-auditor.git
cd audio-qc-manifest-auditor

python3 audit_audio.py sample_audio_manifest.jsonl
```

## License

MIT

