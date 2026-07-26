# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-26 23:15
> 前セッションの要約: TeaCache 品質評価の実験中に CC が「本線が 7.7 倍退行」と誤報し、調査の結果
> 環境退行は無く測定プロトコル不一致だったと判明。派生して対外比較の基準値「論文値 29.00 秒」に
> 一次出典が無いことが発覚し、README / 最終報告を訂正して main にコミットした（36dd4b4）。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**ブランチが 2 系統に分かれている。作業前に必ず確認すること。**

- `main`（現在チェックアウト中）: 対外文書の訂正済み。`36dd4b4` でコミット。`origin/main` より 4 コミット先行（**未 push**）
- `experiment/teacache-quality-eval`: TeaCache 評価ハーネス（`3d3b514`）と DLS-004。**未マージ**。
  TeaCache の残作業はこちら側で行う（`git checkout experiment/teacache-quality-eval`）

### 対外比較の訂正内容（main / 36dd4b4）

| | 基準 | 実測 | 比 | 目標(1.5x) |
|---|---|---|---|---|
| 旧 | 論文値 29.00 秒 | 41.66 秒 | 1.44x | 達成 |
| **新** | **記事 21 秒** | **41.66 秒** | **1.98x** | **未達** |

- 29.00 秒 = 記事の 21 秒（総時間・内訳非公開）をサンプリング単体と読み替え、出所不明のデコード 8.00 秒を加算した構成値だった
- 41.66 秒に conditioning（入力観測エンコード、実測 81.57 秒）は含まれない。`--policy-condition-cache` が warmup の結果を測定 run で再利用するため
- 記事側も内訳非公開だが「生成のみ」と**推定**（確度 7〜8 割。根拠は conditioning 81.57 秒 > 記事総時間 21 秒、および他 3 モードの実測比 1.2〜1.5 倍との整合）
- conditioning 込みの同期総和は 124.91 秒（`result/mainline_full_v4_20260726/`）

### 実行中のバックグラウンド処理（要確認）

**3 モード検証 run が未完了の可能性がある**。`result/verify_3modes_20260726/` に T2I → T2V → I2V を
順次出力する。セッション再開時に完了有無を確認し、公表値と突き合わせる（todo.md Active 最上位）。

### 本線ベンチの再現手順（今後の正 / DLS-005）

```bash
python scripts/run_cosmos_framework_policy_rocm.py --warmup-runs 2 --policy-condition-cache
```

`--policy-condition-cache` 無しでは測定プロトコルが異なり再現しない（322.21 秒という無効値が出る）。
conditioning 込みの 124.91 秒は `--policy-sync-profile`（condition-cache 無し）で測定する。

## 完了済み（今セッション）

- TeaCache 恒等性検証（DLS-004 手順①）: `thresh_0.00` が `baseline_run1` とアクション 160 要素完全一致、
  vision.mp4 の md5 も一致、`forward_calls=30`。**合格**（9 run 中 2 run 完了）
- 「本線 7.7 倍退行」仮説の棄却（DLS-005）: コード・フレームワーク・Docker イメージが当時と同一で、
  フラグ付与により 42.88 秒（記録比 +2.9%）で再現
- 対外比較の基準値是正と conditioning 除外の明示（DLS-006）: README / 最終報告を訂正し main にコミット
- ROCm #5750（Strix Halo 低クロック張り付き）の非該当確認: amd-smi 実測で SCLK 2899MHz（上限 2900MHz）、
  MCLK 1000MHz（400/800/1000 の 3 段構成の最上位）
- und_branch_cache（T2I/I2V）と condition_cache（Policy）の構造的差異を特定 →
  訂正が必要なのは Policy のみと判断
- 長期起動コンテナ 7 個を停止（ユーザー承認済）。RAM 使用 87GiB → 15GiB

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）

## ブロッカー・注意事項

- **未 push**: main が `origin/main` より 4 コミット先行。push はユーザー判断待ち
- **未マージ**: `experiment/teacache-quality-eval` は main の訂正コミット（36dd4b4）を含まない。
  TeaCache 作業を再開する際、必要なら main を取り込む。両ブランチとも `.claude/.dls/active.md` を
  変更しているため、マージ時に衝突する（main 側 = DLS-006/005/003/002/001、experiment 側 = DLS-004/003/002/001）
- **DLS-006 の assumption は未確定**: 「記事の 21 秒は生成のみ」は間接証拠のみの推定。
  記事著者への確認や Cosmos 公式ベンチのスコープ判明で裏取りが必要
- **TeaCache 実験の速度数値は報告しない**（DLS-003 制約）。レポートは品質差のみ
- コンテナ使い捨てのためコールドスタートが 6 月比 7 倍（106 秒 → 730 秒）。1 run あたり 30〜40 分かかる
- Bash 出力が空になる事象が散発。回避策: スクラッチパッドのファイルにリダイレクトして Read

## 関連ファイル

- `.claude/.dls/active.md`（DLS-001〜003, 005, 006。DLS-004 は experiment ブランチ側）
- `.claude/.dls/raw/20260726_chat_baseline_audit_and_scope_estimation.md`（**今セッションの議論ノート。未検証仮説 2 件を含む**）
- `.claude/.dls/raw/20260726_doc_mainline_repro_investigation.md`（退行誤報の調査原本）
- `.claude/.dls/raw/20260726_doc_baseline_and_conditioning_audit.md`（基準値監査の原本）
- `README.md` §2（訂正済み Policy 行と NOTE / IMPORTANT）
- `docs/cosmos3_rocm_policy_optimization_final_report.md`（§1 CAUTION に訂正経緯、§3 結論に conditioning の最適化余地）
- `scripts/run_cosmos_framework_policy_rocm.py`（`--policy-condition-cache` は L234 / L258 / L309-318）
- `result/mainline_repro_v3_20260726/`（42.88 秒、再現確認）、`result/mainline_full_v4_20260726/`（124.91 秒 + sync profile）
- `result/verify_3modes_20260726/`（3 モード検証 run の出力先）
