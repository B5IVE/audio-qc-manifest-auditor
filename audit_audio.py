#!/usr/bin/env python3
"""
audio-qc-manifest-auditor: Audio Captioning, Transcription & Multimodal Manifest QA CLI.

Audits JSONL audio annotation and captioning manifests for schema completeness,
segment duration bounds, chronological/non-overlapping segment boundaries,
empty or low-effort transcripts/captions, and duplicate clip/segment IDs,
emitting a JSON audit report and a Markdown QA summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MIN_DURATION_SEC = 0.25
MAX_DURATION_SEC = 60.0
MIN_TEXT_LEN = 5


@dataclass
class AudioQaIssue:
    line_number: int
    clip_id: str
    severity: str  # ERROR, WARN
    code: str
    message: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit audio captioning/transcription JSONL manifests for QC and timing defects."
    )
    parser.add_argument("manifest_file", type=Path, help="Path to audio manifest .jsonl file")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=MIN_DURATION_SEC,
        help=f"Minimum segment duration in seconds (default: {MIN_DURATION_SEC})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_DURATION_SEC,
        help=f"Maximum segment duration in seconds (default: {MAX_DURATION_SEC})",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("audio_qc_report.json"),
        help="Output JSON report path (default: audio_qc_report.json)",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("audio_qc_summary.md"),
        help="Output Markdown summary path (default: audio_qc_summary.md)",
    )
    args = parser.parse_args()

    if not args.manifest_file.is_file():
        print(f"Error: Manifest file not found: {args.manifest_file}", file=sys.stderr)
        return 2

    issues: list[AudioQaIssue] = []
    seen_segment_keys: dict[tuple[str, str], int] = {}
    segments_by_clip: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    language_counts: Counter[str] = Counter()
    total_records = 0
    valid_segments = 0
    total_duration_sec = 0.0

    with args.manifest_file.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            total_records += 1

            try:
                rec: dict[str, Any] = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(
                    AudioQaIssue(line_num, f"line-{line_num}", "ERROR", "AUD-001", f"Malformed JSON line: {exc}")
                )
                continue

            if not isinstance(rec, dict):
                issues.append(
                    AudioQaIssue(line_num, f"line-{line_num}", "ERROR", "AUD-002", "Record must be a JSON object.")
                )
                continue

            clip_id = str(rec.get("clip_id", "")).strip()
            seg_id = str(rec.get("segment_id", f"seg-{line_num}")).strip()
            text = str(rec.get("transcript", rec.get("caption", ""))).strip()
            lang = str(rec.get("language", "unknown")).strip() or "unknown"

            if not clip_id:
                issues.append(
                    AudioQaIssue(line_num, f"line-{line_num}", "ERROR", "AUD-003", "Missing clip_id.")
                )
                continue

            try:
                start_s = float(rec["start_sec"])
                end_s = float(rec["end_sec"])
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(
                    AudioQaIssue(line_num, clip_id, "ERROR", "AUD-004", f"Missing or non-numeric start_sec/end_sec: {exc}")
                )
                continue

            if start_s < 0.0 or end_s <= start_s:
                issues.append(
                    AudioQaIssue(
                        line_num,
                        clip_id,
                        "ERROR",
                        "TIM-001",
                        f"Invalid segment timestamps ({start_s:.2f}s -> {end_s:.2f}s).",
                    )
                )
                continue

            duration = end_s - start_s
            if duration < args.min_duration:
                issues.append(
                    AudioQaIssue(
                        line_num,
                        clip_id,
                        "WARN",
                        "TIM-002",
                        f"Segment duration {duration:.2f}s below minimum {args.min_duration:.2f}s.",
                    )
                )
            elif duration > args.max_duration:
                issues.append(
                    AudioQaIssue(
                        line_num,
                        clip_id,
                        "WARN",
                        "TIM-003",
                        f"Segment duration {duration:.2f}s exceeds maximum {args.max_duration:.2f}s.",
                    )
                )

            dup_key = (clip_id, seg_id)
            if dup_key in seen_segment_keys:
                issues.append(
                    AudioQaIssue(
                        line_num,
                        clip_id,
                        "WARN",
                        "DUP-SEG",
                        f"Duplicate segment_id '{seg_id}' for clip '{clip_id}' (first seen on line {seen_segment_keys[dup_key]}).",
                    )
                )
            else:
                seen_segment_keys[dup_key] = line_num

            if not text:
                issues.append(
                    AudioQaIssue(
                        line_num,
                        clip_id,
                        "ERROR",
                        "TXT-001",
                        f"Empty transcript/caption on segment '{seg_id}'.",
                    )
                )
            elif len(text) < MIN_TEXT_LEN or text.upper() in {"[INAUDIBLE]", "N/A", "TBD", "..."}:
                issues.append(
                    AudioQaIssue(
                        line_num,
                        clip_id,
                        "WARN",
                        "TXT-002",
                        f"Low-information or placeholder text ({text!r}) on segment '{seg_id}'.",
                    )
                )

            segments_by_clip[clip_id].append((start_s, end_s, line_num))
            language_counts[lang] += 1
            valid_segments += 1
            total_duration_sec += duration

    # Overlap check per clip
    for clip_id, segs in segments_by_clip.items():
        ordered = sorted(segs, key=lambda s: (s[0], s[1]))
        for idx in range(1, len(ordered)):
            prev_start, prev_end, prev_line = ordered[idx - 1]
            curr_start, curr_end, curr_line = ordered[idx]
            if curr_start < prev_end - 1e-6:
                overlap = prev_end - curr_start
                issues.append(
                    AudioQaIssue(
                        curr_line,
                        clip_id,
                        "WARN",
                        "TIM-004",
                        f"Overlap of {overlap:.2f}s with earlier segment on line {prev_line} ({prev_start:.2f}s-{prev_end:.2f}s).",
                    )
                )

    err_count = sum(1 for i in issues if i.severity == "ERROR")
    warn_count = sum(1 for i in issues if i.severity == "WARN")

    report_obj = {
        "metadata": {
            "source_file": str(args.manifest_file),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "total_records": total_records,
            "valid_segments": valid_segments,
            "total_duration_sec": round(total_duration_sec, 3),
            "unique_clips": len(segments_by_clip),
            "language_counts": dict(language_counts),
            "error_count": err_count,
            "warning_count": warn_count,
        },
        "issues": [asdict(i) for i in issues],
    }

    args.json_out.write_text(json.dumps(report_obj, indent=2), encoding="utf-8")

    md_lines = [
        "# Audio Captioning & Transcription Manifest QC Report",
        "",
        f"- **Source File:** `{args.manifest_file.name}` | **Generated (UTC):** `{report_obj['metadata']['generated_utc']}`",
        f"- **Clips:** `{len(segments_by_clip)}` | **Valid Segments:** `{valid_segments}` | **Total Duration:** `{total_duration_sec:.2f}s`",
        f"- **Findings:** `{err_count}` errors, `{warn_count}` warnings",
        "",
        "## QC Findings",
        "",
    ]

    if not issues:
        md_lines.append("_All audio annotation segments passed timing, text, and schema checks._")
    else:
        md_lines.extend(
            [
                "| Line | Clip ID | Severity | Code | Diagnostic Detail |",
                "| :---: | :--- | :---: | :--- | :--- |",
            ]
        )
        for item in issues:
            md_lines.append(
                f"| {item.line_number} | `{item.clip_id}` | **{item.severity}** | `{item.code}` | {item.message} |"
            )

    args.md_out.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    print(
        f"[+] Audio QC: {valid_segments} valid segments across {len(segments_by_clip)} clip(s) "
        f"({total_duration_sec:.2f}s total) — {err_count} errors, {warn_count} warnings."
    )
    print(f"[+] Wrote {args.json_out} and {args.md_out}")

    return 1 if err_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

