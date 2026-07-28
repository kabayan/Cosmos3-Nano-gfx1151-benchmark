# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-28 09:40
> 前セッションの要約: und branch cache の 2 スロット化（DLS-017 採用案 A）を実装・検証し DLS-018 として確定（出力ビット一致のまま T2I 5.25x→2.25x / I2V 11.33x→2.69x に回復）。続く /dls-discuss「チューニング履歴の公平な再分析」は候補 A〜E 提示のまま未決着で中断。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。** 作業ツリーはクリーン（`f188400` + 本コミット）。

**ブランチは 2 系統のまま**（前回から変化なし）:
- `main`（チェックアウト中）: 未 push コミットが origin/main より先行
- `experiment/teacache-quality-eval`（`eed9aa0`）: 未マージ、active.md 衝突あり（**DLS-004 が main の active.md に不在**という帳簿不整合の当事者。議論ノート参照）

### DLS-018（2 スロット化）の確定内容

- 実装: `third_party/diffusers/.../transformer_cosmos3.py` を署名キー LRU 2 スロットに変更
  （third_party クローン内コミット `f829105c7`。third_party は本体 repo の gitignore 対象）
- **イメージ同期済み**: `cosmos3-rocm72-diffusers:local` = `sha256:554e0573ec89...`
  （docker cp + commit 方式。旧 `sha256:eab19ad6eb66...` はロールバック用。
  rebuild は pip 層再実行のネットワーク依存があるため依存変更時のみ）
- 検証: (a) 2 writes / 138 reads / 0 invalidations、(b) T2I jpg / I2V mp4 とも md5 ビット一致、
  (c) T2I 49.633 秒（2.25x）/ I2V 45.622 秒（2.69x）。記録 `result/guidance_2slot_20260728/`
- 公式 guidance 条件の現行倍率: **T2I 2.25x / T2V 2.53x / I2V 2.69x**（3 モードとも価格差 2.0 超過のまま。
  2.0 到達は DLS-017 で非現実的と確定済み、目的化しない）
- T2V への cache 適用（DLS-017 の C 案）は dormant のまま（T2V の 2.53x は cache 無関係の計算量 2 倍が主因）

### 議論の中断状態（/dls-discuss → 未決着）

- topic: 過去のチューニング履歴を公平に再分析し改善点がないか検討
- 原本: `raw/20260728_chat_tuning_history_reanalysis.md`（失敗 3 類型・候補 A〜E・CC 賛否）
- **確定事項なし**（ユーザー選択前に /dls-commit でモード解除）。DLS 未起票（採用判断が無いため）
- 再開手順は todo.md Active 先頭タスクを参照

## 完了済み（今セッション）

- und branch cache 2 スロット化の実装 + イメージ同期 + 検証実測（判定 3 基準すべて合格）
  — 原本 `raw/20260728_doc_und_cache_two_slot_verification.md`、**DLS-018** 起票、コミット `f188400`
- /dls-discuss（チューニング履歴再分析）→ 候補 A〜E 提示まで（未決着で中断、議論ノート保存済み）
- todo.md 整理: 2 スロット化タスク削除（承認済み）、README §2 タスクに新値 2.25x/2.69x を反映

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）
（先頭: 議論候補 A〜E の選択（ユーザー判断）。実装系の先頭候補: README §2 両条件併記の更新）

## ブロッカー・注意事項

- CUDA 参照 run は当分実施不可（ユーザー決定 2026-07-28）。golden 帰属の決着はペンディング
- 記事の実際の guidance は依然未確認（DLS-010 assumption、confidence medium）
- **third_party/diffusers とイメージ /opt/diffusers を乖離させない**（変更したら docker cp + commit で同期、
  依存パッケージ変更時のみ rebuild）。この環境では Bash の grep が無出力になる事象あり（python 検索で代替した）
- 2.0 到達を目的化して計算省略系（TeaCache 等）に手を出すのは DLS-003 でユーザー棄却済み
- 次の Policy run 時にカーネルキャッシュ持ち越し（DLS-015）の golden MSE 帯確認を便乗実施（todo 参照）

## 関連ファイル

- `.claude/.dls/active.md`（DLS-016 / DLS-017 / DLS-018）
- `.claude/.dls/raw/20260728_doc_und_cache_two_slot_verification.md`（実装・検証原本）
- `.claude/.dls/raw/20260728_chat_tuning_history_reanalysis.md`（議論原本、候補 A〜E）
- `result/guidance_2slot_20260728/`（run_commands.sh・summary・出力）
- `third_party/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py`（実装先、クローン内 `f829105c7`）
- `README.md` §2 / §4（更新待ち、todo 参照）
