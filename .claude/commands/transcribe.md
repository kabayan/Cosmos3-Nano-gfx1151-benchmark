---
name: transcribe
description: 指定した WAV 音声ファイルをローカルで日本語に文字起こしする。uvx で whisper-ctranslate2（faster-whisper）を CPU 実行し、入力と同じ場所に同名 .txt を出力する。録音音声の議事録化・文字起こし依頼・「この音声を文字に」と頼まれたときに使う。クラウドにアップロードできない機密音声向け。
---

# /transcribe — ローカル WAV 文字起こし

入力 = `$ARGUMENTS`（文字起こししたい WAV 音声ファイルのパス。必須）。
出力 = 入力 WAV と**同じディレクトリ**に同名の `.txt`（プレーンテキスト）。

ビルド・事前インストール不要。`uvx` が `whisper-ctranslate2`（faster-whisper / CTranslate2）を
都度取得して **CPU で** 実行する。各環境に `uvx` と `ffmpeg` があれば動く。

## 実行手順

1. **引数を検証する**
   - `$ARGUMENTS` が空 → 使い方（`/transcribe <WAVのパス>`）を表示して終了。
   - パスが存在しない / ファイルでない → エラーを表示して終了（処理しない）。

2. **出力先ディレクトリを決める**
   - `OUTDIR = $(dirname "<WAV>")`（ディレクトリ指定が無ければ `.`）。
   - 出力ファイルは `OUTDIR/<basename>.txt` になる（既存があれば上書き）。

3. **文字起こしを実行する**（このコマンドをそのまま Bash で実行）

   ```bash
   uvx whisper-ctranslate2 "<WAV>" \
     --model medium --language ja \
     --device cpu --compute_type int8 --threads 8 \
     --output_format txt \
     --output_dir "<OUTDIR>"
   ```

   - **初回のみ** medium モデル（約 1.5GB）が `~/.cache/huggingface/` にダウンロードされる旨を
     先にユーザーへ伝える（以降はキャッシュ）。
   - CPU 実行のため、長尺音声（例: 1時間）は数分〜実時間程度かかる。長い場合はバックグラウンド実行や
     進捗の見守りを検討する。

4. **結果を報告する**
   - 生成された `OUTDIR/<basename>.txt` の**フルパス**を提示する。
   - 文字起こし結果の**先頭数行**をプレビューとして表示する。

## 固定仕様（変更する場合はこのコマンドを編集）

| 項目 | 値 | 備考 |
|---|---|---|
| エンジン | whisper-ctranslate2（faster-whisper / CTranslate2） | uvx で都度実行・ビルド不要 |
| モデル | medium | 精度と CPU 速度のバランス |
| 言語 | ja（日本語） | |
| デバイス | cpu / compute_type int8 | GPU は使わない（手軽さ優先） |
| 出力形式 | txt のみ | srt/vtt/json 等は出さない |
| 出力先 | 入力 WAV と同じディレクトリ | 同名 `.txt` |

## 注意

- 想定入力は **WAV**。MP3/FLAC 等も内部 ffmpeg 経由で扱えるが、保証対象は WAV。
- 精度・固有名詞が不足する場合は `--model large-v3`、速度を上げたい場合は GPU（`--device cuda`、
  別途 cuDNN 等が必要）が候補。ただし既定は CPU/medium 固定。
