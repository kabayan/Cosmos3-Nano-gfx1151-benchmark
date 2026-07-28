# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-28 02:45
> 前セッションの要約: 決着実験 E4（v1_midtrain 重みでの ROCm golden 照合）を実行し、
> 「v1→v2 の checkpoint 差し替えが golden 不合格の原因」仮説を棄却した（DLS-014）。
> あわせて framework 実行コンテナの定義を repo 化し、毎 run 捨てていたカーネル
> キャッシュを永続化した（DLS-015）。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。** コンテナ・バックグラウンドジョブとも停止確認済み。

**ブランチは 2 系統のまま**:
- `main`（チェックアウト中）: `713386e`、origin/main より 12 コミット先行（未 push）
- `experiment/teacache-quality-eval`（`eed9aa0`）: 未マージ、active.md 衝突あり（前回から変化なし）

### E4 の結論（DLS-014）— 仮説棄却

| run | 重み | golden MSE | 判定 |
|---|---|---:|---|
| v1_ckpt_e4_20260728 | v1_midtrain iter12000 EMA | **0.248372** | FAIL |
| classmethod_policy_framework | v2_midtrain（最適化前） | 0.126471 | FAIL |
| 記事（DGX Spark） | 不明 | 0.013194 | PASS |

- v1 は v2 の約 2 倍悪い。記事が v1 で 0.013 を出していたなら 19 倍乖離する計算になり、
  「公開前 squash 窓での checkpoint 差し替えが原因」という DLS-013 の主線は成立しない
- **ロード自体は正しい**（FAIL の非識別性を一部潰した）: 1165 key 全数が写像・shape/dtype
  一致、ローダは missing_keys で例外を投げるがログに警告なし、出力が v2 比 MSE 0.149132
  動いた（v2 同士の run 間ノイズ 0.001301 の 100 倍超）
- **DLS-013 の「別 checkpoint」判定を訂正**: 全 1165 tensor 比較で相対差の中央値 0.0128、
  43 tensor は bitwise 一致。実態は同一系統の継続学習で、差は生成（moe_gen / diffusion
  expert）経路に集中（action2llm 39.9% / time_embedder 38.8% / action_modality_embed 33.5%）
- 残る交絡: v1 の config.json と現行 yaml の意味論差（qk_norm_for_text 等）。ただし v1 の
  q_norm 重みが学習済みかつ v2 とほぼ同値のため小さいと判断し dormant 化（DLS-014）

### 公開 v2 に policy 精度の基準値が存在しないこと（議論ノート化済み）

- `golden_mse_max = 0.05` は inputs JSON に**データとして書かれているだけ**で、
  framework 内にこれを読む Python コードが 1 行も無い
- `tests/nano_inference_smoke_test.py` は "Smoke-level only (output validity, not
  numeric goldens)" と明記。数値 golden を持つ `launch_regression_test.py` は学習時の
  loss / grad-norm 用で推論出力とは無関係
- HF 公式の action ベンチマーク（`images/benchmark-action-1.png`）は **ID/FD のみ**で
  policy 指標なし
- → 本環境の 12 run は「再現失敗」ではなく **v2 policy 精度の事実上の初回測定**に近い

### カーネルキャッシュ永続化（DLS-015）

- MIOpen / TorchInductor / Triton の 3 種を `scripts/run_cosmos_framework_policy_docker.sh`
  でホストへ持ち越し（`CACHE_DIRS=0` で無効化可）
- プローブ実測（conv3d 2 形状）: 全体 15.75s → 13.48s（−14%）、大 conv 初回 10.76s → 8.52s（−21%）
- ホストの `~/.cache/miopen/3.5.1.5b515cf1bc/gfx1151_20.ukdb` が 188KB → 270KB に増加し
  永続化を確認。**Policy 本体での短縮幅は未測定**

## 完了済み（今セッション）

- E4 一式（重み 30GB 取得 → index 絶対パス修正 → ロードスモーク → golden 照合 run → 採点 →
  全 tensor 比較）→ DLS-014 + 原本
- framework 実行コンテナの repo 化（`docker/cosmos3-rocm72-framework.Dockerfile`、
  `scripts/run_cosmos_framework_policy_docker.sh`）→ DLS-015
- カーネルキャッシュ永続化の実装と効果測定（`scripts/probe_kernel_cache_persistence.py`）
- `run_cosmos_framework_policy_rocm.py` に `--policy-checkpoint-dir` を追加
- 議論ノート `20260728_chat_v2_accuracy_reference_absence.md` を保存し
  DLS-011 / 013 / 014 の sources に追記
- コミット `713386e`、todo.md の完了 2 件を削除

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）

## ブロッカー・注意事項

- **CUDA 参照 run は環境調達が必要でユーザー判断待ち**（AWS g6e.xlarge L40S 等）。
  E4 が仮説を棄却したため、これが残る唯一の決着手段
- **未 push**: main が origin/main より 12 コミット先行
- **次の Policy run では 2 点を必ず確認する**: (1) golden MSE が既存帯 0.126〜0.134 に
  入るか（キャッシュ持ち越しで MIOpen のアルゴリズム選択が変わりうるため。外れたら
  `CACHE_DIRS=0` に戻す）、(2) コールドスタートの短縮幅
- **`--config-file` / `--model-size` は CLI から使えない**: `Training[...]` 注釈のため
  `COSMOS_TRAINING=0`（スクリプトが設定）では tyro から Suppress される。checkpoint の
  差し替えは `--policy-checkpoint-dir` を使う
- README「同一条件」表現の訂正は CUDA 参照 run / guidance 検証の結果待ちで保留（不変）
- `result/` は git 管理外。E4 の成果物は `result/v1_ckpt_e4_20260728/` にのみ存在する
- v1 重み 30GB は `/home/kabayan/workspace/cosmos3_v1_ckpt/`（repo 外、git 管理外）。
  index.json の vision_encoder 絶対パスは相対へ修正済み。再取得するなら
  HF `nvidia/Cosmos3-Nano` revision `35c5cd345afeefabbdebcdc6089f5e5be3402d0f`
- Bash 出力が空になる事象は今セッションも散発。回避策: ファイルへリダイレクトして Read

## 関連ファイル

- `.claude/.dls/active.md`（DLS-001〜003, 005〜015。DLS-004 は experiment ブランチ側）
- `.claude/.dls/raw/20260728_doc_e4_v1_checkpoint_golden_verification.md`（E4 原本）
- `.claude/.dls/raw/20260728_chat_v2_accuracy_reference_absence.md`（議論ノート）
- `scripts/run_cosmos_framework_policy_docker.sh` / `docker/cosmos3-rocm72-framework.Dockerfile`
- `scripts/run_cosmos_framework_policy_rocm.py`（`--policy-checkpoint-dir`）
- `scripts/check_policy_golden_mse.py`（採点）、`scripts/probe_kernel_cache_persistence.py`
- `tasks/todo.md`（CUDA 参照 run が最上位）
