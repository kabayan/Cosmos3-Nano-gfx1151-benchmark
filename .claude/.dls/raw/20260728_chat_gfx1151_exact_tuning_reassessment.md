# 議論ログ: gfx1151 精度不変チューニング再評価

- 日付: 2026-07-28
- 種別: chat / discussion
- 参加: ユーザー / Codex
- 発端: ユーザー要求「過去のチューニング履歴を公平に判断し、高速化ができないかを判定。精度を下げることは原則不可」

## 議論の目的と判定基準

過去の採用・棄却判断を追認するのではなく、実測済み・未検証・外部更新待ちを分離し、
生成条件、モデル重み、dtype、演算内容を変えない追加高速化の余地を判定した。

検証時の非劣化基準は、決定的な diffusers 経路では同一入力・seed の出力 hash 一致、
run-to-run 非決定性がある Policy 経路では既存ノイズ帯と golden MSE 帯からの非劣化とする。

## 過去履歴の公平な再評価

### 再試行しない候補

- TeaCache、INT8/SageAttention、sparse attention、steps/frame/guidance 削減:
  計算内容または生成条件を変え、DLS-003 の制約に反する
- vLLM/PagedAttention: batch 1・固定長・毎 step 全再計算の DiT に利益源がない（DLS-002）
- channels_last_3d: warm 実測で baseline より 5 倍以上遅い
- deeper TunableOp: I2V で約 0.08 秒しか改善せず、実質効果なし
- Stream-K: 安全版は transformer 71.7 秒に対し 78.2 秒、別構成は実行失敗
- CFG cond/uncond バッチ化: 実運用トークン帯の利得約 1.9%（DLS-017）
- 非同期 decode: 同期総和を短縮せず、比較指標だけを変える

### 未検証で残る候補

1. T2V で既存の厳密 und branch cache を有効化し、速度と出力 hash を比較する。
   実装済み機構を使う低コスト検証だが、T2V の und 系列は小さく効果は限定的と見込む。
2. PyTorch 2.13 + AOTriton 0.12b を隔離環境で比較する。
   gfx1151 の正式経路化や compiler 修正の効果は未測定だが、劇的改善の根拠はない。
3. gfx1151 専用 AOTriton tuning database の upstream 完了後に head_dim=128 を再評価する。

## Web調査で判明した新証拠

### AOTriton

- PR #200（2026-07-13 merge）は gfx1151 で平均約 61%、最大 97% の attn_fwd 改善を報告するが、
  head_dim=64 限定。Cosmos3-Nano の head_dim=128 には直接適用できない。
- PR #205（2026-07-27 Draft）は gfx1151 の全 head dimension・dtype・系列長・構成を対象とし、
  head_dim=128 を初めて含む。Level 1 correctness test は進行中、性能比較は TBD。
- PR #203 は gfx1151 部分 DB を gfx1100 fallback DB と合成する仕組みだが未マージ。
  中間検証では 272 有効構成中 50 構成で invalid argument が発生し、runtime 検証が継続中。
- AOTriton 0.13b は maintenance release で、compiler と tuning database の変更はない。
  0.12b から 0.13b へ上げるだけの性能施策は成立しない。
- PyTorch 2.13 は AOTriton 0.12b を同梱し、gfx1151 を experimental から正式経路へ移したが、
  PR #205 の full tuning database は含まない。

参照:
- https://github.com/ROCm/aotriton/pull/200
- https://github.com/ROCm/aotriton/pull/203
- https://github.com/ROCm/aotriton/pull/205
- https://github.com/ROCm/aotriton/releases/tag/0.13b
- https://github.com/pytorch/pytorch/releases/tag/v2.13.0

### hipBLASLt / TheRock

- TheRock issue #2591 で AMD が gfx1151/gfx1201 の MIOpen・hipBLASLt 実 workload shape log を収集中。
  gfx1151 の shape tuning が完成段階でないことを示す。
- gfx1151 で hipBLASLt を強制しても hipBLAS より遅い事例があり、backend の無条件強制は不可。
- TheRock 7.11 が stock ROCm 7.2 より大幅に速いという利用報告はあるが、AMD 側の追試では
  電源 profile の効果が workload により約 2〜4%またはゼロ。Cosmos3への一般化はできない。

参照:
- https://github.com/ROCm/TheRock/issues/2591
- https://github.com/ROCm/ROCm/issues/5643
- https://github.com/ROCm/TheRock/discussions/2845

### Kokoro-FastAPI issue #454 と MIOpen cache

- 同 issue は `MIOPEN_FIND_MODE=2` と事前 warmup でプロセス再起動時の探索を回避する事例。
- 公式仕様では `2 = FAST`。FindDb hit 時は保存済み結果を使うが、miss 時は immediate fallback を使い、
  起動短縮と引き換えに GPU 定常性能が下がる可能性がある。
- Cosmos3 は DLS-015 で MIOpen User FindDb、TorchInductor、Triton cache を既にホストへ永続化済み。
  `MIOPEN_FIND_MODE` の強制は solver 選択条件を変えるため本線では採用しない。

参照:
- https://github.com/remsky/Kokoro-FastAPI/issues/454
- https://rocm.docs.amd.com/projects/MIOpen/en/develop/reference/env_variables.html

## 候補と評定

| 候補 | 評定 | 理由 |
|---|---|---|
| T2V 厳密 cache + PyTorch 2.13 の限定検証 | 賛成 | 精度・計算内容を維持でき、未検証部分を低〜中コストで閉じられる |
| PR #203/#205 完了後の head_dim=128 probe | 留保付き賛成 | Cosmos3へ直接効く可能性があるが、現時点は correctness・統合・性能評価が未完 |
| Draft PRを現在の本線へ手動導入 | 反対 | upstream 自身が runtime 互換性問題を検証中で、精度不変条件を満たす根拠がない |
| 広範な再チューニング campaign | 反対 | 既棄却案の再演になり、20%以上の改善を支える新証拠がない |
| `MIOPEN_FIND_MODE=2` の無条件常用 | 反対 | miss 時の fallback により solver と定常性能が変わりうる |
| 何もしない | 留保 | 残候補の実測改善が 5%未満なら費用対効果上合理的 |

## 結論

現行 stable stack で、精度・計算内容を維持した 20%以上の追加改善を期待する根拠はない。
ただし「追加高速化不能」と断定するのも強すぎる。T2V 厳密 cache と PyTorch 2.13 の限定検証、
および upstream PR #203/#205 完了後の head_dim=128 probe に範囲を絞る。

限定検証がいずれも 5%未満なら、現状スタックでは追加チューニングの費用対効果なしとして停止する。
Draft AOTriton DB と MIOpen FAST mode は、correctness・solver 同一性を確認できるまで本線に入れない。
