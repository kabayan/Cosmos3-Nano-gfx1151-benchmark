# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-28
> 前セッションの要約: `/dls-discuss` で gfx1151 の過去チューニングを精度不変条件のもと再評価し、最新 upstream 情報を調査して DLS-020 に決着した（`240a9fb`）。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。** 議論モードは `/dls-commit` で解除済み。

- HEAD: `240a9fb`（DLS-020、gfx1151 精度不変チューニング再評価）
- main は origin/main より先行した未 push コミットを保持
- `experiment/teacache-quality-eval` は未マージ（DLS-004の品質評価トラック）
- `.agents/`、`.codex/`、`AGENTS.md`、`agents/` は今回の議論前から存在する未追跡ファイル。
  チューニング議論とは別件のため `240a9fb` には含めず、そのまま保持した

### DLS-020 の決着

- 精度不変の意味を、生成条件・モデル重み・dtype・演算内容を維持し、現行出力に対する
  非劣化を実証することとした
- 現行 stable stack で20%以上の追加改善を期待する根拠はないが、追加高速化不能とも断定しない
- 過去に棄却済みの TeaCache本線導入、INT8/SageAttention、vLLM/PagedAttention、
  channels_last_3d、deeper TunableOp、Stream-K、CFGバッチ化は再試行しない
- 未検証対象は、T2V厳密und branch cache、PyTorch 2.13隔離比較、upstream完了後の
  gfx1151/head_dim=128 AOTriton probeに限定
- 限定検証が5%未満なら追加campaignを停止する

### 最新 upstream 調査

- AOTriton PR #205（2026-07-27 Draft）はgfx1151の全head dimensionを対象とし、
  Cosmos3-Nanoのhead_dim=128を含む。Level 1 correctness進行中、性能TBD
- 部分DB統合のPR #203も未マージで、runtime互換性問題を検証中
- PR #200の平均約61%・最大97%改善はhead_dim=64限定で、Cosmos3へ直接適用不可
- AOTriton 0.13bはcompiler/tuning DB変更なし。単独upgradeは性能施策にならない
- PyTorch 2.13はAOTriton 0.12bでgfx1151を正式経路化したが、PR #205は未収録
- Kokoro-FastAPI #454の `MIOPEN_FIND_MODE=2` はFAST mode。FindDb miss時に
  immediate fallbackを使い定常性能が下がりうるため、本線には採用しない
- DLS-015のMIOpen/Inductor/Triton cache永続化は維持する

## 完了済み（今セッション）

- 過去DLS・raw・性能レポート・実測resultを再読し、採用済み／棄却済み／未検証を分離
- gfx1151関連のROCm、TheRock、PyTorch、AOTriton、hipBLASLt、MIOpen情報を一次情報中心に調査
- AOTriton PR #203/#205という新しいhead_dim=128再評価条件を特定
- Kokoro-FastAPI #454をローカルDLS-015と照合し、cold-start対策は既に導入済みと判定
- 議論構造を `raw/20260728_chat_gfx1151_exact_tuning_reassessment.md` に保存
- DLS-020と限定検証タスクを追加し、`240a9fb` でコミット

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体はtodo.mdに一元化）

## ブロッカー・注意事項

- AOTriton PR #203/#205は未マージ・検証中。Draft版を本線へ手動導入しない
- diffusers経路の非劣化は同一入力・seedの出力hash、Policy経路は既存run-to-runノイズ帯と
  golden MSE帯で判定する
- `MIOPEN_FIND_MODE=2` は単なるcache読込指定ではなくsolver選択方針を変えるFAST mode
- PyTorch 2.13は既存環境を上書きせず、隔離コンテナで比較する
- `playwright-cli` は環境に未導入。今回のURL本文取得は内蔵Webへフォールバックした
- CUDA参照runは当分実施不可（既存のユーザー決定）
- third_party/diffusersと実行イメージ `/opt/diffusers` を乖離させない

## 関連ファイル

- `.claude/.dls/active.md`（DLS-020）
- `.claude/.dls/raw/20260728_chat_gfx1151_exact_tuning_reassessment.md`（議論原本）
- `tasks/todo.md`（限定検証タスク）
- `scripts/benchmark_classmethod_article_t2v_i2v_rocm.py`（T2V厳密cache候補）
- `scripts/run_cosmos_framework_policy_docker.sh`（DLS-015 cache永続化）
- `docs/cosmos3_rocm_further_speedup_reassessment_20260726.md`（従来の再評価）
