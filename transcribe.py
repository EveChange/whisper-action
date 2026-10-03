#!/usr/bin/env python3
"""
transcribe.py - Speech-to-Text transcriber using faster-whisper and yt-dlp.
Supports multiple YouTube URLs, model selection (including Large V3 Turbo),
quantization (int8, float16), and outputs to SRT, TXT, and CSV.
"""

import argparse
import csv
import os
import re
import sys
import tempfile
from typing import List, Dict, Any

# Map user-friendly model names to faster-whisper model identifiers
MODEL_MAP = {
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
    "large-v3": "large-v3",
    "large-v2": "large-v2",
    "large": "large-v3",
    "medium": "medium",
    "small": "small",
    "base": "base",
    "tiny": "tiny",
}


def sanitize_filename(name: str) -> str:
    """Sanitize video title for use as a filesystem filename."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip()
    return name if name else "transcript"


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm."""
    millis = int(round(seconds * 1000))
    hours = millis // 3600000
    millis %= 3600000
    minutes = millis // 60000
    millis %= 60000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments: List[Any], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp_srt(seg.start)
            end = format_timestamp_srt(seg.end)
            text = seg.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def write_txt(segments: List[Any], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.text.strip() + "\n")


def write_csv(segments: List[Any], filepath: str):
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "text"])
        for seg in segments:
            writer.writerow([round(seg.start, 2), round(seg.end, 2), seg.text.strip()])


def parse_urls(raw_urls: str) -> List[str]:
    """Parse comma, newline, or space separated URLs."""
    raw = raw_urls.replace(",", "\n")
    urls = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            for part in line.split():
                part = part.strip()
                if part.startswith("http://") or part.startswith("https://"):
                    urls.append(part)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def download_audio(url: str, temp_dir: str) -> Dict[str, str]:
    """Download audio from a YouTube URL using yt-dlp."""
    import yt_dlp

    ydl_opts = {
        "format": "ba/b",
        "outtmpl": os.path.join(temp_dir, "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "audio")
        # yt-dlp creates a .wav file after postprocessing
        expected_wav = os.path.join(temp_dir, f"{title}.wav")
        if not os.path.exists(expected_wav):
            # Find any .wav in temp_dir
            for f in os.listdir(temp_dir):
                if f.endswith(".wav"):
                    expected_wav = os.path.join(temp_dir, f)
                    break
        return {"title": title, "audio_path": expected_wav}


def resolve_audio_files(path_expr: str) -> List[str]:
    """Resolve a path, glob expression, directory, or comma-separated list into audio filepaths."""
    import glob
    SUPPORTED_EXTS = {".ogg", ".wav", ".mp3", ".m4a", ".aac", ".flac", ".wma", ".opus", ".mp4", ".mkv", ".webm"}
    files = []
    parts = [p.strip() for p in path_expr.replace("\n", ",").split(",") if p.strip()]
    for part in parts:
        if os.path.isdir(part):
            for root, _, filenames in os.walk(part):
                for f in sorted(filenames):
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                        files.append(os.path.join(root, f))
        else:
            matches = glob.glob(part)
            if matches:
                for m in sorted(matches):
                    if os.path.isfile(m):
                        files.append(m)
            elif os.path.isfile(part):
                files.append(part)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for f in files:
        abs_p = os.path.abspath(f)
        if abs_p not in seen:
            seen.add(abs_p)
            result.append(f)
    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe YouTube videos or local audio with faster-whisper")
    parser.add_argument("--urls", type=str, default="", help="YouTube video URL(s)")
    parser.add_argument("--audio-path", type=str, default="", help="Local audio file path")
    parser.add_argument("--model", type=str, default="large-v3-turbo", help="Whisper model name")
    parser.add_argument("--output-dir", type=str, default="youtube", help="Output directory")
    parser.add_argument("--output-filename", type=str, default="", help="Custom output filename without extension")
    parser.add_argument("--format", type=str, default="txt,srt,csv", help="Comma-separated output formats (txt,srt,csv,all)")
    parser.add_argument("--translate", action="store_true", help="Translate source language to English")
    parser.add_argument("--compute-type", type=str, default="int8", help="Quantization compute type (int8, float16, etc.)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda)")
    parser.add_argument("--language", type=str, default=None, help="Spoken language code (optional)")

    args = parser.parse_args()

    urls = parse_urls(args.urls) if args.urls else []
    if not urls and not args.audio_path:
        print("Error: No URLs or --audio-path provided.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve model
    model_key = args.model.lower().strip()
    model_id = MODEL_MAP.get(model_key, args.model)
    print(f"Loading Whisper model: {model_id} (device={args.device}, compute_type={args.compute_type})...")

    from faster_whisper import WhisperModel
    model = WhisperModel(model_id, device=args.device, compute_type=args.compute_type)

    requested_formats = [f.strip().lower() for f in args.format.split(",")]
    if "all" in requested_formats:
        requested_formats = ["srt", "txt", "csv"]

    task = "translate" if args.translate else "transcribe"
    success_count = 0

    # Process local audio file(s) if provided
    if args.audio_path:
        audio_files = resolve_audio_files(args.audio_path)
        if not audio_files:
            print(f"Error: No audio files found matching: {args.audio_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(audio_files)} audio file(s) to transcribe.")
        for idx, audio_file in enumerate(audio_files, start=1):
            base_name = args.output_filename if (args.output_filename and len(audio_files) == 1) else os.path.splitext(os.path.basename(audio_file))[0]
            audio_dir = os.path.dirname(audio_file)
            target_dir = args.output_dir if (args.output_dir and args.output_dir != "youtube") else (audio_dir or ".")
            os.makedirs(target_dir, exist_ok=True)
            print("\n" + "=" * 60)
            print(f"[{idx}/{len(audio_files)}] Processing audio file: {audio_file}")
            print("=" * 60)
            segments_iter, info = model.transcribe(audio_file, task=task, language=args.language, vad_filter=True)
            print(f"Detected language: '{info.language}' (probability: {info.language_probability:.2f})")
            segments = []
            for seg in segments_iter:
                segments.append(seg)
                print(f"[{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}] {seg.text.strip()}")

            base_path = os.path.join(target_dir, base_name)
            if "srt" in requested_formats:
                write_srt(segments, f"{base_path}.srt")
                print(f"Saved: {base_path}.srt")
            if "txt" in requested_formats:
                write_txt(segments, f"{base_path}.txt")
                print(f"Saved: {base_path}.txt")
            if "csv" in requested_formats:
                write_csv(segments, f"{base_path}.csv")
                print(f"Saved: {base_path}.csv")
            success_count += 1



    for idx, url in enumerate(urls, start=1):
        print("\n" + "=" * 60)
        print(f"[{idx}/{len(urls)}] Processing YouTube URL: {url}")
        print("=" * 60)

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                print("Downloading audio with yt-dlp...")
                download_res = download_audio(url, temp_dir)
                title = download_res["title"]
                audio_file = download_res["audio_path"]
                safe_title = sanitize_filename(title)
                print(f"Video Title: {title}")
                print(f"Audio file extracted: {audio_file}")

                print(f"Transcribing audio with task='{task}'...")
                segments_iter, info = model.transcribe(
                    audio_file,
                    task=task,
                    language=args.language,
                    vad_filter=True,
                )

                print(f"Detected language: '{info.language}' (probability: {info.language_probability:.2f})")
                print("--- Transcription Progress ---")

                segments = []
                for seg in segments_iter:
                    segments.append(seg)
                    print(f"[{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}] {seg.text.strip()}")

                base_path = os.path.join(args.output_dir, safe_title)

                if "srt" in requested_formats:
                    srt_path = f"{base_path}.srt"
                    write_srt(segments, srt_path)
                    print(f"Saved: {srt_path}")

                if "txt" in requested_formats:
                    txt_path = f"{base_path}.txt"
                    write_txt(segments, txt_path)
                    print(f"Saved: {txt_path}")

                if "csv" in requested_formats:
                    csv_path = f"{base_path}.csv"
                    write_csv(segments, csv_path)
                    print(f"Saved: {csv_path}")

                success_count += 1
                print(f"Finished transcribing: {title}")

            except Exception as e:
                print(f"Failed to process {url}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Batch transcription complete. Succeeded: {success_count}/{len(urls)}")
    print("=" * 60)

    if success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
