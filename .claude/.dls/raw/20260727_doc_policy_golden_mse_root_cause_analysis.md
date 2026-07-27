# 原本: golden MSE 不合格（DLS-011)の原因分析 — 消去法による絞り込み

- 日付: 2026-07-27
- 種別: doc（分析ログ）
- 文脈: ユーザー指示「そもそも精度が下回る原因をコードも含めて分析」。GPU 実行なし、
  コード読解 + 既存出力の後処理のみで実施。DLS-011 の続き。

## 1. 新しい実測知見: 誤差の構造は「計画全体の時間前倒し」

MSE(pred[t], golden[t+k]) を k = -3..+3 で計算（mainline_full_v4）:

| k | -2 | -1 | **0** | **+1** | **+2** | +3 |
|---|---|---|---|---|---|---|
| 全次元 | 0.323 | 0.217 | **0.128** | **0.109** | 0.141 | 0.248 |
| dim9 グリッパー | 1.151 | 0.806 | 0.507 | 0.273 | **0.0071** | 0.219 |
| dim0 x並進 | 0.304 | 0.193 | 0.074 | **0.032** | 0.083 | 0.190 |
| dim2 z並進 | 0.497 | 0.290 | 0.129 | **0.071** | 0.157 | 0.341 |

- グリッパーは **+2 ステップずらすと MSE 0.507 → 0.0071（71 分の 1）**。
  golden は step 8 で閉じ、本環境は step 6 で閉じる。
- 並進 3 次元は +1 ずらしで最良。
- **結論**: 値のノイズではなく、**同じ動作計画を 1〜2 ステップ早回しで実行**している。
  ランダムな劣化なら全 k で悪化するはずで、こうはならない。

## 2. 消去法（すべて実測 / blob 照合ベース）

| 原因候補 | 判定 | 根拠 |
|---|---|---|
| 速度最適化の副作用 | **除外** | 最適化前 6/1 相当 run が既に 0.126（DLS-011） |
| 数値精度モード（fp16 説） | **除外** | run ログ実測 `precision torch.bfloat16` = 記事と同一 |
| NATTEN sparsity 差 | **除外** | config 実測 two_way / natten_parameter_list None。sparsity は three_way 限定 |
| 観測フレーム選択・前処理 | **除外** | `read_media_frames` = `torchvision.io.read_video` → **先頭 17 フレーム切出しのみ**（`inference/vision.py:178-190`、`action.py build_action_batch` L101-105 も先頭切詰め）。リサンプリング無し。H.264 デコードは仕様上決定的。リサイズではなく reflection パディング |
| 初期ノイズの環境差 | **除外** | `misc.arch_invariant_rand`（`utils/misc.py:145`）= **NumPy RandomState(seed) の CPU 生成** → 名前どおりアーキ不変。vision / action / sound 全ノイズがこの経路（`omni_mot_model.py:1672,1698,1728`）。seed=0 固定。golden・記事・本環境は同一初期ノイズ |
| チェックポイント版差 | **除外** | HF キャッシュに 3 snapshot（138d071c=6/1 期, 03c14e74=6/6, 411f42a8=7/9 現行 main）。**差分は README.md と modular_model_index.json のみ**。model.safetensors.index.json 以下の重み blob は 3 版で同一。記事（検証版）〜現在で重み不変 |
| attention の因果マスク整合（causal_type 無視） | **ほぼ除外** | TopLeft/BottomRight の差は seqlen_q ≠ seqlen_kv でのみ発生（`frontend.py` docstring）。本経路は packed self-attention で Q/KV 長一致 |
| seed=0 の falsy 取り違え | **除外（傍証）** | fallback seed が発火していれば run ごとに独立ノイズになり run 間 MSE ≤0.0086 と矛盾 |

## 3. 残った容疑者: ROCm スタックの順伝播カーネル数値差

入力・初期ノイズ・重み・グラフ意味論がすべて一致している以上、
軌道の分岐は**順伝播の数値差**からしか生まれない。本環境固有の演算経路:

1. **attention（最有力）**: 本モデルの全 attention は
   `_sdpa_varlen_fallback`（run script L1051-1052 で `attention_frontend.attention` を無条件差替え）
   → `F.scaled_dot_product_attention` / `torch.ops.aten._flash_attention_forward`
   → gfx1151 では `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` の **AOTriton 実験カーネル**。
   実験扱い = 精度検証が公式に通っていない。attention は 74k トークン級で 30 ステップ × 全層通過し、
   誤差の増幅点として最大。
2. **MIOpen bf16 conv（VAE encode）**: conditioning latent frame[0] を作る経路。
   サンプリング入力の条件側に直接効く。
3. **hipBLASLt bf16 GEMM**: 一般に cuBLAS と同等精度だが、TunableOp 選択カーネルの検証は無い。

規模の傍証: 記事（GB10）は golden（NVIDIA 生成、生成環境非公開）に対し 0.013。
異なる CUDA アーキ間の数値差では軌道が 0.013 に収まる = 正しいスタックでは軌道は
数値差に頑健。本環境の 0.128 はその 10 倍で、「per-op の数値差が CUDA 系より一桁大きい
演算がどこかにある」ことを示唆する。序盤ステップ（高シグマ、計画の大勢が決まる領域）での
velocity の差が計画のタイミングを 1〜2 ステップ動かし、以降は自己整合的に生成されるため、
出力は「壊れた動画」ではなく「早回しの正常な計画」になる — §1 の構造と整合する。

## 4. 切り分け実験の提案（未実施、要ユーザー承認 — GPU 実行）

感度分析: 疑わしい演算を 1 つずつ高精度化し、golden MSE の応答を見る。
CUDA 参照が手元に無い以上、これが唯一の実証手段。

- **E1（最優先・実装 30 分）**: `_sdpa_varlen_fallback` を fp32 アップキャスト + SDPA math backend
  （セグメントループ）に差し替えて 1 run。fallback は自前コードなので monkeypatch 済み地点を
  書き換えるだけ。MSE が 0.05 未満に落ちれば attention カーネル精度が主因と確定
- **E2**: VAE encode のみ fp32 化して 1 run（conditioning 経路の分離）
- **E3**: モデル全体 fp32（重み 15.17B×4B ≈ 61GB、VRAM 120GB で可行）。E1/E2 が外れた場合の総当たり
- 判定基準: MSE < 0.05（公式合格）まで落ちれば主因確定。0.128 → 0.06〜0.09 程度の部分改善なら
  複合要因として E2/E3 を重ねる

## 5. 本分析で訂正した本セッション内の誤り

- 「観測ウィンドウのフレーム選択ズレが時間シフトの原因かもしれない」→ コード実測で否定
  （先頭 17 フレーム固定で選択の自由度が無い）
- 「初期ノイズが ROCm RNG 由来で別サンプルになっている可能性」→ 実装読解で否定（CPU NumPy 生成）
