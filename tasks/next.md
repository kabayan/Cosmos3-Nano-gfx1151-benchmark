# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-27 14:30
> 前セッションの要約: `/dls-discuss 精度深堀り` から tokenizer pin の法医学調査を実施（DLS-013）。
> tokenizer 代替仮説を棄却し、真犯人候補として **HF checkpoint の v1→v2 差し替え**（公開前
> squash 窓 5/13〜5/31）を発見。golden はデータセット実測（不変）と判明し、CUDA 不要の決着実験
> E4（v1 重み ROCm 照合）を定義した。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。** GPU 実行なしの調査セッション（HF/GitHub API +
ローカル HF キャッシュ照合のみ）。

**ブランチは 2 系統のまま**:
- `main`（チェックアウト中）: origin/main より先行（未 push）
- `experiment/teacache-quality-eval`（`eed9aa0`）: 未マージ、active.md 衝突あり（前回から変化なし）

### 今セッションで確定した事実（DLS-013、詳細は原本参照）

1. **tokenizer pin 無罪**: pin の正体は `Cosmos3-Nano.yaml` L185 の VLM processor revision
   `a18b727665f0dd03bc032229a6acb47ba11dc4cb`。404 の理由は HF main の 2026-06-01 Super-squash。
   processor が読む全ファイルは pre-squash 5/13（branch `spectralflight/shim`）〜現行 main で
   oid/blob 完全一致 → **代替 revision 使用は原因ではない**
2. **v1→v2 checkpoint 差し替え発見**: shim の checkpoint.json は `cosmos3_ga_16bm8b_v1_midtrain`
   iter12000 EMA、公開 framework は初日から `v2_midtrain` 前提。tensor 名全面リネーム
   （814 key 中共通 2）+ バイト照合で layer0 layernorm が corr 0.90・96% 相違 = **別 checkpoint**。
   本環境の全 run は v2（ローカル 3 snapshot すべて新 key 形式）
3. **golden の正体訂正**: `golden_action_path` == `action_path`（cosmos-dependencies pin
   `2b17a2413bd8` の bridge_20260501_0.json、16×10 実測アクション）。golden 側ドリフトは構造的に
   不可能、**動いたのはモデル側だけ**。「golden は 2026-05 内部コード生成」は誤った枠組みだった
4. **記事公開日 = 2026-06-01**（squash・pin→main と同日）。検証実施は 6/1 以前で v1 時代に
   跨がる可能性。公開 v2 checkpoint の golden 合格は誰も検証していない可能性が高い
   （CI は numeric golden 対象外）

### 議論の確定事項（/dls-discuss「元実装で精度は出ている？」）

- (a) 最適化前の本環境コード: 不合格 0.126471（確定） / (b) 記事環境: 第三者による公式 golden
  照合で合格 0.013194 / (c) 現公開コード + CUDA: 未検証
- 記事の 0.013194 は「自己申告」ではなく元リポジトリの公式 golden・公式メトリクスとの照合実測
  （golden_mse_max 0.05 への言及で確証）

## 完了済み（今セッション）

- tokenizer pin 法医学調査一式（pin 特定 → squash 発見 → shim branch 発掘 → oid/blob 照合 →
  tensor バイト照合 → golden 正体特定 → 記事日付特定）→ DLS-013 + 原本
- todo.md 更新: E4（v1 重み ROCm 照合）を最上位に追加、CUDA 参照 run を後続に格下げ

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）

## ブロッカー・注意事項

- **E4 実行はユーザー承認待ち**（30GB DL + GPU 実行 + tensor リネーム移植）。
  FAIL はロード誤りと識別不能（PASS のみが情報を持つ）— 撤退ラインを事前設定すること
- **未 push**: main が origin/main より先行
- README「同一条件」表現の訂正は E4 / CUDA run の帰属確定まで保留が安全（前回から不変）
- a18b727 の内容直接照合は永久に不可能（squash で消滅）。二重反転（a18b727 だけ tokenizer が
  違った可能性）は原理的に排除できないが、前後（5/13 と 6/1）で一致しており蓋然性は低い
- Bash 出力が空になる事象が今セッションも散発（git status / date で再現）。
  回避策: 再実行 or ファイルにリダイレクトして Read
- 記事 HTML・tensor 照合スクリプト（cmp_tensor.py）・index json 類はセッション scratchpad に
  あり消える。再現手段は原本 §8 に記録済み

## 関連ファイル

- `.claude/.dls/active.md`（DLS-001〜003, 005〜013。DLS-004 は experiment ブランチ側）
- `.claude/.dls/raw/20260727_doc_tokenizer_pin_forensics_and_v1v2_checkpoint_swap.md`（今回の原本）
- `.claude/.dls/raw/20260727_doc_policy_golden_mse_precision_sweep.md`（E1〜E3、前提知見)
- `tasks/todo.md`（E4 タスク最上位）
- HF: `nvidia/Cosmos3-Nano` branch `spectralflight/shim` rev `35c5cd345`（v1 重み、E4 の取得元）
