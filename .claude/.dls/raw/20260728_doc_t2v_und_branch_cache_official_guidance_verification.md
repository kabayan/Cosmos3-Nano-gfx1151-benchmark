# T2V 公式 guidance 条件 und branch cache 検証

- 日付: 2026-07-28
- 実行者: Codex（`/dls-continue` で `tasks/todo.md` Active 最上位を執行）
- 目的: DLS-020 の限定検証 (1)。T2V 公式 guidance 条件で既存の厳密 und branch cache を有効化し、速度と出力完全一致を確認する

## 影響範囲分析

- 分類 A（DLS-020 で予定済みの実測・採否判断）
- 実装変更なし。既存の Diffusers-native 2 スロット cache を使用
- 影響対象: DLS-016（公式 guidance T2V 値）、DLS-017（dormant 候補 C）、DLS-018（T2V 効果は限定的との当時の見通し）、DLS-020（限定検証の完了）
- DLS-003 の精度・計算内容不変条件に従い、同一入力・seed の MP4 byte 一致を合格条件とした

## 実行前照合

- HEAD: `9544378`
- image: `cosmos3-rocm72-diffusers:local` (`sha256:554e0573ec89...`)
- image 内 `/opt/diffusers/.../transformer_cosmos3.py` と `third_party/diffusers/.../transformer_cosmos3.py` の SHA-256 はともに `c814b68710455142535c60a36a8f462ebead3f681663ce726e7c58720bce9208`
- cache API の存在をコンテナ内 import で確認: `enable_und_branch_cache = True`
- baseline: `result/guidance_official_20260728/t2v/summary.json`
  - guidance 6.0、35 steps、256x448、24 requested frames、seed 202
  - measured total 55.561 秒
  - transformer 50.251 秒 / 70 calls
  - MP4 SHA-256 `d086071dc8808359cf743322233910b510dbee9e0332cd4c5f224f4ca9908373`

## 実行条件

baseline と同じ Docker image、AOTriton experimental、TunableOp 読み込み、VAE warmup、mode warmup 1 + measured 1 を使用し、差分を `--und-branch-cache` のみに限定した。

成果物:

- `result/t2v_und_cache_official_20260728/summary.json`
- `result/t2v_und_cache_official_20260728/run.log`
- `result/t2v_und_cache_official_20260728/article_t2v_red_cube_256p_24f_s35.mp4`

`result/` は gitignore 対象のため、再生成不能な判断原本として本ファイルに数値・hash・条件を転記する。

## 結果

| 指標 | cache 無効 | cache 有効 | 差 |
|---|---:|---:|---:|
| measured total | 55.561 秒 | 40.806 秒 | -26.56% / 1.362x speedup |
| transformer | 50.251 秒 | 35.552 秒 | -29.25% / 1.413x speedup |
| VAE decode | 4.090 秒 | 4.028 秒 | ほぼ不変 |
| transformer calls | 70 | 70 | 計算ステップ・CFG 呼び出し数は不変 |

cache stats（warmup + measured 合計）:

```json
{
  "enabled": true,
  "transformer_calls": 140,
  "write_calls": 2,
  "read_calls": 138,
  "invalidations": 0,
  "cached_slots": 2,
  "cached_layers": 72,
  "cache_gib": 0.139
}
```

出力同一性:

- baseline / cache 有効の SHA-256 はともに `d086071dc8808359cf743322233910b510dbee9e0332cd4c5f224f4ca9908373`
- MD5 はともに `92876a5e8dfcde36802187b0b5b5edb6`
- `cmp -s` exit 0（byte 完全一致）

## 判定

- 合格。DLS-020 の停止線 5% を大幅に超え、精度・生成条件・演算内容を変えずに 26.6% 短縮した
- T2V の公式 guidance 最適化経路として採用する
- 記事値 22 秒に対する倍率は 55.561 / 22 = 2.53x から 40.806 / 22 = 1.85x に改善し、価格差 2.0 基準の内側に入る
- PyTorch 2.13 隔離比較と upstream AOTriton head_dim=128 再評価は別の限定検証として継続する
