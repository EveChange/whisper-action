#!/bin/sh

set -eu

model=${INPUT_MODEL:-"large-v3-turbo-q5_0"}

[ -d models ] || mkdir -p models

# Check if model is already an existing file path
if [ -f "$model" ]; then
  export INPUT_MODEL="$model"
else
  # Normalize model name: lowercase, trim, replace spaces with dashes
  m=$(echo "$model" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
  # Strip leading ggml- and trailing .bin if present
  m=$(echo "$m" | sed -e 's/^ggml-//' -e 's/\.bin$//')
  # Normalize quant aliases: q5-0 -> q5_0, q8-0 -> q8_0, q5-1 -> q5_1
  m=$(echo "$m" | sed -e 's/-q5-0/-q5_0/' -e 's/-q8-0/-q8_0/' -e 's/-q5-1/-q5_1/')

  # Map common aliases
  case "$m" in
    turbo) m="large-v3-turbo" ;;
    turbo-q5_0|turbo-q5) m="large-v3-turbo-q5_0" ;;
    turbo-q8_0|turbo-q8) m="large-v3-turbo-q8_0" ;;
    large-turbo) m="large-v3-turbo" ;;
    large-turbo-q5_0) m="large-v3-turbo-q5_0" ;;
    large-turbo-q8_0) m="large-v3-turbo-q8_0" ;;
    large) m="large-v1" ;;
  esac

  filename="ggml-${m}.bin"

  if [ ! -f "models/${filename}" ]; then
    echo "===> Downloading Whisper model: ${filename} from Hugging Face..."
    if ! curl -f -LJ "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${filename}" \
        --output "models/${filename}"; then
      echo "Error: Failed to download model '${filename}' from Hugging Face." >&2
      echo "Please verify the model name or quantization variant." >&2
      exit 1
    fi
  fi

  export INPUT_MODEL="models/${filename}"
fi

raw_urls="${INPUT_YOUTUBE_URL:-${INPUT_YOUTUBE_URLS:-}}"

if [ -n "$raw_urls" ]; then
  # Split by commas, newlines, and spaces to support multiple URLs
  echo "$raw_urls" | tr ',' '\n' | tr ' ' '\n' | tr -d '\r' | while IFS= read -r line || [ -n "$line" ]; do
    url=$(echo "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    [ -z "$url" ] && continue
    echo "=========================================================="
    echo "===> Transcribing YouTube URL: $url"
    echo "=========================================================="
    if ! INPUT_YOUTUBE_URL="$url" /bin/go-whisper "$@"; then
      echo "Failed to transcribe: $url" >&2
    fi
  done
else
  /bin/go-whisper "$@"
fi

