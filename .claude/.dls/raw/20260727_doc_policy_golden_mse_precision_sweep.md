# 原本: fp32 感度実験 E1〜E3 — golden MSE 不合格の数値精度仮説を全棄却

- 日付: 2026-07-27
- 種別: doc（実験ログ）
- 文脈: DLS-011（golden MSE 不合格）の原因切り分け。分析原本
  `20260727_doc_policy_golden_mse_root_cause_analysis.md` の §4 提案をユーザー承認のうえ実行。

## 1. 実験設計

疑わしい演算を段階的に fp32 化し、golden MSE の応答を見る感度分析。
実装は `scripts/run_cosmos_framework_policy_rocm.py` に既定 OFF のオプトインフラグ 3 本:

| フラグ | 内容 |
|---|---|
| `--policy-attn-fp32-math` (E1) | 全 attention を fp32 アップキャスト + SDPA math backend（AOTriton 完全迂回、GQA は KV 明示展開、varlen はセグメント毎の等長 self-attention） |
| `--policy-vae-encode-fp32` (E2) | Wan VAE の重み・scale・dtype を fp32 化（MIOpen bf16 conv 迂回。encode/decode 両方） |
| `--policy-model-fp32` (E3) | `load_model_config_dict` の戻りで `config.precision = "float32"`（GEMM 含む transformer 全体。E1 併用必須 = flash 系 fp32 非対応） |

全 run: 同一入力（公式サンプル、seed 0）、warmup なし、`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`。
フラグ有効化はログマーカー（[E1]/[E2]/[E3]）と `OmniMoTModel: precision torch.float32`（E3）、
サンプリング速度低下（E1: 1.17→2.55 s/it）で確認済み。

## 2. 結果

| run | 構成 | golden MSE | flip steps | 対 bf16 ベースライン MSE |
|---|---|---|---|---|
| ベースライン ×11 | bf16 + AOTriton | 0.126〜0.134 | [6,7] | ≤0.0086（run 間） |
| `attn_fp32_e1_20260727` | E1 | 0.134018 | [6,7] | 0.0025〜0.0080 |
| `vae_fp32_e2_20260727` | E1̸+E2（E2 のみ） | 0.131866 | [6,7] | 0.0020（対 E1 0.00015） |
| `model_fp32_e3_20260727` | E1+E2+E3（全系 fp32） | 0.126543 | [6,7] | **0.000478** |

**すべての精度構成で出力は run-to-run ノイズ帯（≤0.0086）の内側。** グリッパー flip は
[6,7] で不変。全系 fp32 でも bf16 と実質同一（0.0005）。

## 3. 結論

1. **数値精度仮説の全棄却**: attention（AOTriton 実験カーネル / fallback 意味論）、
   MIOpen bf16 conv（VAE）、hipBLASLt bf16 GEMM（全 transformer）のいずれも原因ではない。
2. **軌道は数値摂動に対し完全に頑健**: 本環境は「step 6 でグリッパーを閉じる計画」に
   決定論的に収束する。カオス増幅説も棄却（大きな精度摂動でも計画が動かない）。
3. 副産物: `_sdpa_varlen_fallback`（自作 fallback）の意味論の正しさが fp32 数値基準との
   一致で実証された。
4. 残る仮説は**入力・コード版の意味論差**のみ:
   - golden は 2026-05-01 の内部コードで生成（公開履歴は 5/31 開始でそれ以前は見えない）
   - 記事環境は「正式リリース前の検証版」+ tokenizer pin `a18b727…`（HF から消滅、内容照合不能）
   - 公開ウィンドウ 5/31→6/13 の diff は policy 経路に意味論変更なし（別原本 §済）
   - 公開リポジトリの CI は action golden を検証しておらず（smoke test は
     「numeric goldens は対象外」と明記）、**公開コードが May golden を再現する保証は最初から無い**

## 4. 決着に必要な実験（未実施・環境調達が必要）

**CUDA 参照 run**: 本プロジェクトと同一の公開コード（b3967db）+ 同一入力を CUDA GPU で 1 本実行し
golden MSE を採点する（AWS g6e L40S 等）。
- CUDA が PASS（≈0.013）→ ROCm 側に精度以外の意味論差が残存（要再調査）
- CUDA が FAIL（≈0.13）→ 公開コード自体が May golden を再現しない。ROCm 移植は完全に無罪で、
  差は上流の版差と確定

外部記事の代用可否も調査済み: RTX 5090 記事（zenn）は policy 未検証、L40S 記事
（DevelopersIO 2026-07-03）は本文に到達できず未確認。

## 5. 実行記録

- E1: 起動 09:50 頃 → 完了 10:22（サンプリング 2.55 s/it）
- E2: 〜11:00 完了
- E3: 〜11:40 完了（precision torch.float32 をログ確認）
- 判定スクリプトは公式 `compute_action_mse` と同一式（golden 照合原本 §1 参照）
