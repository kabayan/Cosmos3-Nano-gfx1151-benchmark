# 原本: guidance 公式デフォルト実測（速度・品質）と記事 per-step チャートの発見

- 日付: 2026-07-28
- 種別: doc（実験ログ）
- 文脈: /dls-discuss「v1/v2 の精度・速度の乖離と許容範囲」から派生したユーザー決定
  (1) CUDA 参照 run は当分実施不可、(2) 精度は素の実測提示に凍結、
  (3) 対外比較の合否軸を「対記事倍率が価格差 2.0 以内か」に置く、
  (4) DLS-010 dormant の guidance 条件不一致リスクを公式デフォルト実測で潰す（速度・精度とも）。

## 1. 実行条件

DLS-007 v3 再現プロトコルと完全同一（イメージ `cosmos3-rocm72-diffusers:local`、
`aotriton_tuned` env、TunableOp 表 `tunableop_results0.csv` md5 `0e28785495b0b8d1e002ce7ed759337b`
を実行前に確認、pip install なし）。変更点は `--guidance` のみ:

- T2V / I2V: 1.0 → **6.0**（公式 `defaults/text2video・image2video/sample_args.json` 準拠）
- T2I: 1.0 → **4.0**（公式 `defaults/text2image/sample_args.json` 準拠。steps は記事準拠 35 を維持、公式デフォルト 50 ではない）

実行スクリプト: `result/guidance_official_20260728/run_commands.sh`（コマンド全文を保存）。
3 run 逐次、全て exit 0（04:16〜04:56 UTC）。

## 2. 速度結果

| mode | 記事 (DGX Spark) | guidance 1.0（README 公表） | 倍率 | 公式 guidance（今回） | 倍率 | 価格差 2.0 判定 |
|---|---:|---:|---:|---:|---:|---|
| T2I (4.0) | 22 秒 | 27.136 | 1.23x | **115.589** | **5.25x** | 超過 |
| T2V (6.0) | 22 秒 | 32.165 | 1.46x | **55.561** | **2.53x** | 超過 |
| I2V (6.0) | 17 秒 | 25.045 | 1.47x | **192.521** | **11.33x** | 超過 |

stage 内訳（measured run）:

| mode | transformer_forward | calls | s/call | vae_decode | unattributed |
|---|---:|---:|---:|---:|---:|
| T2V @6.0 | 50.251 | 70 | 0.718 | 4.090 | 1.201 |
| I2V @6.0 | 187.087 | 70 | 2.673 | 4.161 | 1.251 |
| T2I @4.0 | 113.254 | 70 | 1.618 | 1.762 | — |

参考（guidance 1.0、DLS-007 v3）: T2V 26.836s/35calls=0.767 s/call、
I2V 19.560s/70calls=0.279 s/call（cache read 主体）、T2I 24.938s/70calls=0.356 s/call（同）。

## 3. 機構の確定

1. **CFG は逐次 2 回 forward**（`pipeline_cosmos3_omni.py` L1597-1667。バッチ倍増ではない）。
   GEMM 形状不変のため TunableOp 表は有効なまま。T2V（cache 不使用構成）の s/call が
   0.767→0.718 と不変であることが直接証拠。T2V の悪化 = 純粋な呼び出し回数 2 倍。
2. **und branch cache は CFG 下で全スラッシュ**（`transformer_cosmos3.py` L871-900、
   単一スロット署名キャッシュ）。cond/uncond が毎回署名不一致 →
   **140 calls / 140 writes / 0 reads / 140 invalidations**（T2I・I2V とも実測）。
   キャッシュ効果ゼロ化 + write オーバーヘッドで、T2I は s/call が 0.356→1.618（4.5 倍）、
   I2V は 0.279→2.673（9.6 倍）。stale 読み出しは起きない設計（出力の正しさは保たれる）。
3. I2V が T2I よりさらに悪いのは und 枝に画像トークンが載り再計算コストが大きいため。

## 4. 品質（精度側）

- T2I: guidance 4.0 の出力（`result/guidance_official_20260728/t2i/*.jpg`）は 1.0
  （`result/verify_3modes_v3_20260726/t2i/*.jpg`）と品質クラスが別物。アーム形状・
  グリッパー・積み木・ラップトップが整合的に生成され、1.0 の構造崩壊が解消。
  guidance は品質に決定的で、「同一条件」主張には guidance 一致が必須と確認。
- T2V/I2V の mp4 も両条件で保存済み（数値 golden は T2V/I2V 系に存在しない。
  golden_psnr_min があるのは forward dynamics / policy 系のみ）。

## 5. 記事 per-step MSE チャートの発見（既存 DLS 未記録の一次情報）

記事 HTML から本文画像 2 点を取得（scratchpad art1/art2）:

- art2 = **「Cosmos3 Policy Model — 公式 golden action との Per-step MSE」チャート**
  （URL: devio2024-media .../v1779425546/2026/05/22/x4znbyx1n1okduwey2qr.png）
  - 記事側（DGX Spark）も **step 6・7 で per-step MSE ≈ 0.099 / 0.100 と
    公式しきい値 0.05 を局所超過**（チャート内で赤表示 + しきい値線明示）
  - 全体 MSE 0.0132 は「step 6-7 以外ほぼゼロ」という構造での合格
  - 画像アップロードパスが **2026/05/22** → 記事の検証実施は 5/22 以前と絞れる
    （squash 窓 5/13〜5/31 の内側。DLS-013 の時系列に追加）
- art1 = Policy の条件動画 Frame0 / 生成 Frame0・8・16 の比較図（bridge タスク）

## 6. 本環境の per-step MSE（同じ軸での比較、`scripts/check_policy_golden_mse.py` の golden 使用）

| step | 記事（チャート読取） | v2 (mainline_full_v4, 0.128000) | v1 (E4, 0.248372) |
|---:|---:|---:|---:|
| 0 | ≈0 | 0.0015 | 0.0070 |
| 1 | ≈0 | 0.0374 | 0.0820 |
| 2 | ≈0 | 0.0386 | 0.0243 |
| 3 | ≈0 | 0.2677 | 0.2575 |
| 4 | ≈0 | 0.1282 | 0.1023 |
| 5 | ≈0 | 0.0437 | 0.0545 |
| **6** | **0.099** | **0.4823** | **0.4481** |
| **7** | **0.100** | **0.4459** | **0.4372** |
| 8 | ≈0 | 0.0237 | 0.0417 |
| 9 | ≈0 | 0.0442 | 0.1004 |
| 10 | ≈0 | 0.0709 | 0.0860 |
| 11 | ≈0 | 0.1493 | 0.4633 |
| 12 | ≈0 | 0.0826 | 0.4081 |
| 13 | ≈0 | 0.0236 | 0.1230 |
| 14 | ≈0 | 0.1216 | 0.7054 |
| 15 | ≈0 | 0.0867 | 0.6330 |

- 本環境 v2 の最大逸脱は**記事と同じ step 6-7**（グリッパータイミング帯、値は約 4.5 倍）
  で、そこに広帯域誤差（step 3,4,10-12,14,15 等）が加算される構造
- v1 は後半（step 11-15）が大きく発散し、全域で v2 より悪い（DLS-014 と整合）
- 「記事側も golden と完全一致ではない（2 step は per-step しきい値超過）」は
  精度の素の提示（ユーザー決定）に併記する価値がある一次情報

## 7. 留保

1. 記事が実際に公式デフォルト guidance で実行したかは依然未確認
   （DLS-010 assumption、confidence medium のまま）。ただし本測定で 1.0 / 公式値の
   両条件が実測済みになり、assumption がどちらに倒れても対応する実測値が存在する
2. negative prompt はスクリプト既存値（T2V: 短文ハードコード、I2V: 記事アセット、
   T2I: 短文ハードコード）で、公式 `neg_prompts.json`（長文構造化）と異なる。
   uncond 側系列長が公式条件より短く、速度はわずかに本環境有利側の誤差
3. 各条件 measured 1 run。ただし DLS-007 で同プロトコルの再現性 ±0.6% を確認済み、
   measured run の unattributed は 1.2 秒でクリーン

## 8. 帰結（チューニング候補、未採否）

公式 guidance 条件で価格差 2.0 以内に入れるための候補（採否は /dls-plan で扱う）:

- und branch cache の 2 スロット化（cond/uncond 各 1）: 両 context とも step 不変のため
  厳密キャッシュのまま read 化を回復できる。概算で T2I ≈ 2×0.356×70+α ≈ 52 秒（2.4x）、
  I2V ≈ 2×0.279×70+α ≈ 45 秒（2.6x）— **回復してもまだ 2.0 超**の見込み
- T2V への und cache 適用（現行構成は未使用）
- 何もしない（guidance 1.0 の値を条件明記のうえ主提示に残す）

T2V のクリーンな実測（cache 無関係）が 2.53x であることから、CFG 有効条件での
ハードウェア素の比は約 2.5 倍と推定され、厳密計算の範囲で 2.0 以内into到達には
per-call 20% 以上の追加短縮が必要になる。

## 9. 検証根拠（再現手段）

- `bash result/guidance_official_20260728/run_commands.sh`
- summary: `result/guidance_official_20260728/{t2v,i2v}/summary.json`、`t2i/article_t2i_summary.json`
- per-step MSE: `scripts/check_policy_golden_mse.py` の golden キャッシュ +
  `sample_outputs.json`（本 doc §6 の計算はホスト python 標準ライブラリのみ）
- 記事画像: `curl -s https://dev.classmethod.jp/articles/dgx-spark-cosmos3-omni-world-model-policy/`
  → 本文 img 2 点（2026/05/22 アップロード）
