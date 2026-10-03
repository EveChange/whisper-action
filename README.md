# whisper-action

Speech-to-Text using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) and [yt-dlp](https://github.com/yt-dlp/yt-dlp) for [GitHub Actions](https://github.com/features/actions). High-performance, reliable inference of OpenAI Whisper automatic speech recognition (ASR) models, including **Large V3 Turbo**.

## Inputs variables

See [action.yml](./action.yml) for more detailed information.

| Variable         | Description                                                  | Default |
|------------------|--------------------------------------------------------------|---------|
| model            | Whisper model (`large-v3-turbo`, `large-v3`, `large-v2`, `medium`, `small`, `base`, `tiny`) | `large-v3-turbo` |
| youtube_urls     | YouTube Video URL(s) (supports single or multiple URLs).    |         |
| audio_path       | Local audio path (e.g. `./testdata/audio.wav`).              |         |
| output_folder    | Output folder.                                               | `youtube` |
| output_format    | Output format, supports `txt`, `srt`, `csv`, or `all`.       | `txt,srt,csv` |
| output_filename  | Custom output filename without extension.                    |         |
| translate        | Translate from source language to English (`true`/`false`).  | `false` |


## Usage

Donwload Youtube video and transcript it.

```yaml
jobs:
  youtube-eng-video:
    name: transcript english video
    runs-on: ubuntu-latest
    steps:
    - name: checkout
      uses: actions/checkout@v3

    - name: speech to text
      uses: appleboy/whisper-action@v0.1.1
      with:
        model: small
        youtube_url: https://www.youtube.com/watch?v=pTCxXZh6VyE
        output_format: srt
        output_folder: youtube
        print_segment: true
        debug: true

    - name: git push changes
      uses: appleboy/git-push-action@v0.0.2
      with:
        branch: main
        commit: true
        commit_message: "[skip ci] Upload changes"
        remote: git@github.com:appleboy/whisper-action.git
        ssh_key: ${{ secrets.DEPLOY_KEY }}
        rebase: true
```

See the output file in youtube folder.
