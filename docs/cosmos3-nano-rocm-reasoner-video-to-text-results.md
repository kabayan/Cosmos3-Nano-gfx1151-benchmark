# Cosmos3-Nano ROCm Reasoner Video-to-Text Results

実施日: 2026-06-01

対象: `nvidia/Cosmos3-Nano`

## 結論

ROCm/gfx1151 上の vLLM Reasoner 経路で video-to-text が成功した。

使用動画:

```text
https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4
```

## Video

Downloaded file:

```text
result/reasoner/car-detection.mp4
```

Metadata:

```text
format: MP4
codec: H.264
resolution: 768x432
fps: 12.5
frames: 377
duration: 30.16 sec
size: 2.7 MiB
audio: AAC
```

## 起動時の重要設定

元動画を `--media-io-kwargs '{"video": {"num_frames": -1}}'` で全フレーム投入すると、入力長が 10372 tokens になり、今回の `max_model_len=8192` を超えて 400 error になった。

成功時は server 起動時に動画フレーム数を 16 に制限した。

```text
--media-io-kwargs '{"video": {"num_frames": 16}}'
--max-model-len 8192
--max-num-seqs 1
--dtype float16
--enforce-eager
--gpu-memory-utilization 0.75
--skip-mm-profiling
--mm-processor-kwargs '{"max_pixels": 262144}'
```

## Request

Video URL sent to server:

```text
file:///workspace/result/reasoner/car-detection.mp4
```

Prompt:

```text
Describe the video. Focus on the visible scene, vehicles, motion, and any notable events. Answer in concise Japanese.
```

## Response

Elapsed:

```text
14.171 sec
```

Output:

```text
駐車場の上空から撮影された映像で、白い車が画面中央に現れ、右方向へと移動していきます。その後、赤い車と青い車が画面下部から上部へと移動し、白い車が画面左上に現れます。
```

## Output Files

```text
result/reasoner/car-detection.mp4
result/reasoner/car-detection-4fps.mp4
result/reasoner/video_to_text_response.txt
result/reasoner/video_to_text_response.json
result/reasoner/video_models.json
result/reasoner/cosmos3_reasoner_video_server.log
```

## Notes

- 4 fps に再エンコードした `car-detection-4fps.mp4` も作成したが、server 起動時の `num_frames=-1` では入力長が 10372 tokens のままだった。
- `--media-io-kwargs '{"video": {"num_frames": 16}}'` を server 起動時に指定するのが有効だった。
- `cosmos3-rocm-reasoner` container は video test 用の 16-frame sampling 設定で起動中。

