# conditioning プローブ実測 — 「189 フレームエンコード」主張の棄却と訂正（原本）

- 日時: 2026-07-27 00:25 〜 00:47
- 検証者: CC
- 前提原本: `.claude/.dls/raw/20260726_doc_conditioning_full_frame_encode_verification.md`（同日の静的解析。**§2-3 の結論は誤り**）
- 実行: `scratchpad/probe_conditioning_inner.py`（`OmniMoTModel.encode` を包み、初回呼び出しで実測して中断）
- 出力: `result/conditioning_probe_20260727/`

## 1. 実測されたテンソル形状（決定的証拠）

```json
{"conditioning_probe": {
  "input_shape":        [1, 3, 17, 544, 736],
  "full_latent_shape":  [1, 48,  5,  34,  46],
  "slice_input_shape":  [1, 3,  5, 544, 736],
  "slice_latent_shape": [1, 48,  2,  34,  46],
  "dtype": "torch.bfloat16"
}}
```

VAE encode に渡るのは **17 ピクセルフレーム**であり、189 ではない。
プローブの `sample_args.json` は mainline run（`result/mainline_full_v4_20260726/`）と
`vision_path` / `action_path` / `output_dir` の 3 つ以外**完全一致**（`num_frames: 189` も同じ）なので、
条件差ではなく前回の読み違いである。

## 2. 何を読み違えたか

`num_frames: 189` は解決済み sample_args に確かに存在するが、**policy モードでは vision テンソルの
長さに使われない**。`inference.py` L513-517 で action 系モード（forward_dynamics /
inverse_dynamics / policy）は早期 return する:

```python
if sample_args.model_mode.is_action:
    from cosmos_framework.inference.action import get_action_sample_data
    return get_action_sample_data(...)
```

前回追った `build_conditioned_video_batch`（`vision.py` L106、`num_frames` でパディングする関数）は
L540 以降にあり、**policy モードでは到達しない**。T2V / I2V 等の条件付き動画生成の経路だった。

policy が実際に通るのは `inference/action.py`:

- `get_action_sample_data` L163: `read_media_frames(vision_path, max_frames=action_chunk_size + 1)`
  → 入力観測動画から **17 フレーム**だけ読む（`action_chunk_size=16`）
- `build_action_batch` L94-99: `target_frames = action_chunk_size + 1 = 17`。
  17 未満なら最終フレームで pad、17 超なら切り詰め
- 544x736 は `reflection_pad_to_target` による解像度パディング後の値

「81.57 秒 ÷ 0.43 秒/フレーム ≈ 189 フレーム」という前セッションの見積もりと `num_frames: 189` の
一致は**偶然**だった。実際は 81.575 秒 ÷ 17 フレーム ≈ 4.8 秒/フレーム。

## 3. 無駄仮説そのものは別の形で残る

`build_sequence_plan_from_mode`（`data/vfm/action/transforms.py` L296-299）:

```python
# image2video/forward_dynamics/policy: first frame is clean (conditioning)
if mode in ["image2video", "forward_dynamics", "policy"]:
    condition_frame_indexes_vision = [0]
```

policy の条件付けは **latent frame [0] のみ**。
（解決済み sample_args の `condition_frame_indexes_vision: [0,1]` は CLI 既定値
`DEFAULT_CONDITION_FRAME_INDEXES_VISION[VIDEO]` であり、policy 経路はこれを消費しない。
実際に model へ渡るのは `build_action_batch` が作る sequence_plan の `[0]`。）

policy は「first frame を条件に action と video の**両方**を生成する」モード
（transforms.py L258 のドキュメント文字列）。したがって推論時、
`noise = cond_mask * x0 + (1 - cond_mask) * pure_noise`（`omni_mot_model.py` L1677）において
latent frame 1〜4 は `cond_mask = 0` となり、**エンコード結果は捨てられ shape だけが使われる**。

因果 VAE（`(n-1)*4+1` 対応）なら latent frame 0 に必要なのはピクセルフレーム 0 の 1 枚のみ。
→ **17 フレーム中 16 フレーム（約 94%）のエンコードが推論では捨てられている**可能性が残る。

前回の主張との差分:

| | 前回（誤） | 今回（実測） |
|---|---|---|
| encode 入力 | 189 フレーム | **17 フレーム** |
| 経路 | `build_conditioned_video_batch` | **`build_action_batch`** |
| 条件 latent | [0, 1] | **[0]** |
| 捨てられる割合 | 184/189 = 97.4% | **16/17 = 94%（未実証）** |

## 4. 時間の実測は失敗（無効値）

```json
{"full_encode_seconds": 716.17, "slice_encode_seconds": 1105.04, "speedup": 0.65}
```

17 フレームのエンコードに 716 秒、5 フレームに 1105 秒。**フレーム数の少ない方が遅い**。
mainline の実測 `get_data_and_condition_sync` は 81.575 秒なので、両方とも桁違いに大きい。

原因は MIOpen のカーネル探索。コンテナ使い捨て + MIOpen ユーザー DB のホストマウント無しのため
（`tasks/todo.md` Active の既知事項、コールドスタート 6 月比 7 倍）、初回 conv 形状ごとに
探索が走る。スライス版は 17→5 フレームで conv 形状が変わり、**もう一度フル探索**が走った。

したがって **「81.6 秒 → 約 2.2 秒」という削減見積もりは依然として未実証の外挿**である。
本プローブでは検証できなかった。

## 5. 因果性（latent 一致）も判定不能

```json
{"latent01_bitwise_equal": false, "latent01_max_abs_diff": 0.03125, "latent01_mean_abs": 0.519}
```

bf16、平均絶対値 0.519 に対し最大差 0.03125（= 2^-5）。単純な丸め誤差より大きい。
ただし本結果は判定に使えない:

1. 比較したのは latent frame **[0:2]**。§3 のとおり policy で意味を持つのは **[0] のみ**
2. 17 フレーム版と 5 フレーム版で conv 形状が異なり、MIOpen が別アルゴリズムを選んだ可能性がある
   （§4 のとおり両者とも独立にカーネル探索が走った）。bf16 での累積順序差は
   この規模の差を生じうる

## 6. 次に必要な検証（再設計）

1. 比較対象を latent frame **[0:1]** に変更する
2. 入力を **ピクセルフレーム 1 枚**（`state[:, :, :1]`）にする。policy の条件付けに必要な最小単位
3. MIOpen 探索の影響を排除する。各エンコードを 2 回実行して 2 回目を採る、
   または `MIOPEN_USER_DB_PATH` をホストにマウントして探索結果を持ち越す
   （`tasks/todo.md` Active の「MIOpen カーネルキャッシュのホストマウント」と同じ施策）
4. `cond_mask` の非ゼロインデックスを実測でダンプし、§3 の静的読解（[0] のみ）を裏付ける

## 7. 教訓

静的コード読解だけで「97.4% が無駄」と結論を出しかけた。分岐の取り違え（action 系モードの
早期 return を見落とし、到達しない関数を追った）が原因。
本セッションではこれで 3 度目の同型の誤り（v1: イメージ / diffusers 不一致、
v2: TunableOp 表の欠落、今回: コード経路の取り違え）であり、
いずれも**実測で分岐が確定するまで結論を出さない**ことで防げた。
