# 議論ログ: さらなる高速化調査のスコープ決定（/dls-plan）

- 日付: 2026-07-26
- 参加: ユーザー / CC (Sonnet 5)
- 発端: ユーザー要求「/dls-plan 現在までのチューニング履歴を参照してさらなる高速化できないかを調査してドキュメント化」

## want

gfx1151 での Cosmos3 推論について、既存チューニング履歴を踏まえた残余高速化余地の
調査結果が 1 つのドキュメントとして残っている。

## DRY チェック結果（docs/ 66 ファイルより）

- DLS active.md / archive.md は本時点で未作成（本件が初エントリ）
- 実施済み（final report 2026-06-14、総実行時間 41.66s / 目標 43.5s 達成）:
  - Graph Break 全解消（.tolist() 排除）、_get_velocity 同期排除
  - HIP Graphs + torch.compile、TunableOp GEMM、AOTriton SDPA
  - Condition Cache（303→148s）、VAE upsample_3 部分 compile（95.8→9.4s）
  - MIOpen Workspace 制限緩和（rocm-libraries local-opt）
- 棄却済み仮説:
  - channels_last_3d 化（warm 測定で baseline より 5 倍以上遅い、
    docs/cosmos3-rocm-v2_3-policy-decoder-block-and-velocity-rocprof-results.md）
  - compute-bound 説（真因はメモリ帯域 280GB/s 飽和。74k トークン FlashAttention の
    KV 再ロードが支配的。1.127s/step は物理限界と結論、
    docs/cosmos3_rocm_performance_limit_reassessment.md）
- 既出未実施（dormant 相当）:
  - vLLM-cosmos3 統合（v2.5 提案書記載、final report に適用記録なし）
  - 非同期 VAE デコード / steps・解像度削減（同一条件制約により対象外）
  - Linear/Sparse Attention、シーケンス長削減（limit reassessment が列挙のみ）

## 候補と採否

| 案 | 内容 | 採否 |
|---|---|---|
| A | 同一条件（30steps/640x480/17f・同期総和）維持のまま、環境更新・未完 vLLM 統合・attention 帯域削減策を再調査しドキュメント化 | **採用（ユーザー承認）** |
| B | 制約緩和込み全方位ロードマップ（量子化・sparse attention・非同期化・steps 削減） | dormant: final report §3 と重複、同一条件比較というプロジェクト当初価値と矛盾 |
| A+B | A 主軸 + B 付録の統合ドキュメント | dormant: 主軸がぼける、B 部分は既出情報の再掲になる |
| C | 何もしない（YAGNI） | dormant: 最終報告から約 6 週間経過し ROCm/AOTriton 等の更新が「物理限界」結論の前提を変えた可能性が未反映になる |

## 反対視点（議論モード必須分）

案 A への反対:「限界と結論した直後に再調査するのは結論を信用していないのと同じでは」
→ 反論: 結論の根拠は当時のソフトウェアスタック（ROCm 7.2 / 当時の AOTriton）に依存。
gfx1151 向け最適化は ROCm 側で活発に進行中であり、時間経過そのものが再調査の正当な根拠。
