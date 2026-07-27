# 次のセッションへの引き継ぎ

> 作成日時: 2026-07-27 13:11
> 前セッションの要約: 4 モードの対外比較条件を監査し（DLS-010: guidance 1.0 vs 公式デフォルト
> 4.0/6.0 の不一致を発見）、Policy 出力を公式 golden と照合して全 run 不合格を確定（DLS-011:
> MSE 0.126〜0.134 vs 合格 0.05）。fp32 感度実験 E1〜E3 で ROCm 数値精度仮説を全棄却し（DLS-012）、
> 原因を上流の版差・入力差に絞った。コミット済み（`7215021`）。

DLS-123: 本ファイルは **文脈・状態の運搬** に専念する。タスク本体は `tasks/todo.md` の
`Active` セクションに一元化する。

---

## 現在の状態

**実行中のバックグラウンド run は無い。** ワーキングツリーはクリーン、`7215021` が最新。

**ブランチは 2 系統のまま**:
- `main`（チェックアウト中）: `origin/main` より **9 コミット先行（未 push）**
- `experiment/teacache-quality-eval`（`eed9aa0`）: 未マージ。両ブランチが active.md を変更して
  おりマージ時衝突（main 側 = DLS-012〜005, 003〜001 / experiment 側 = DLS-004, 003〜001）

### 今セッションで確定した 3 つの大きな事実

1. **guidance 不一致（DLS-010）**: T2I/T2V/I2V の公表倍率 1.23x/1.46x/1.47x は guidance 1.0
   （CFG 無効）で測定。公式デフォルトは T2I 4.0 / T2V・I2V 6.0 で、`guidance != 1.0` は
   1 ステップ 2 順伝播（`omni_mot_model.py` L2367-2369）。記事が公式デフォルトで実行していた
   場合、約半分の計算量での比較になり倍率は成立しない。**Policy のみ公式デフォルト 1.0 で一致**。
   guidance 1.0 の採用根拠を記した文書は repo に存在しない（無根拠に引き継がれていた）
2. **golden MSE 全 run 不合格（DLS-011）**: 判定は `python scripts/check_policy_golden_mse.py`
   （今回追加、全 run 一括採点）。記事は 0.013194 で PASS。誤差の構造は「計画全体の 1〜2
   ステップ前倒し」（グリッパーは +2 シフトで MSE 0.507→0.0071、golden は step 8 / 本環境は
   step 6 で閉じる）。**最適化前の 6/1 run から一貫して 0.126 なので速度最適化は無罪**
3. **数値精度は原因ではない（DLS-012）**: E1（attention fp32 math）/ E2（+VAE fp32）/
   E3（全系 fp32）すべて出力が run 間ノイズ帯（≤0.0086）内で不動。フラグは
   `--policy-attn-fp32-math` / `--policy-vae-encode-fp32` / `--policy-model-fp32`（E3 は E1 併用必須）。
   残る仮説は上流版差のみ: golden は 2026-05-01 内部コード生成、記事は「検証版」+ 消滅した
   tokenizer pin `a18b727`（HF 404）、公開ウィンドウ 5/31→6/13 の diff は policy 経路に意味論
   変更なし、**公開リポジトリの CI は action golden を検証していない**

### 記事側条件の確定度（DLS-010 監査結果）

- 記事は公式サンプル JSON をそのまま実行（golden_mse_max 0.05 への言及で確証）
- T2I: 960×960/35steps/22s 明記。T2V: 256p/24f/12fps/**35steps（まとめ行に明記**、前半の
  「記載なし」報告は WebFetch 要約の誤りで curl 原文取得により訂正）/22s。I2V: 17s のみで
  条件は全部本プロジェクトの仮定。Policy: 640×480×17f/16×10/21s
- 入力データ: T2I=公式 t2i.json、T2V=**自作プロンプト**（公式 t2v サンプルは別内容のため不可避）、
  I2V=公式だが framework の i2v.json が指す robot_153.jpg とは**別ファイル**、Policy=完全公式
- 記事原文の取得は curl 直接（`playwright-cli` 未インストール）。WebFetch 要約は 2 回
  食い違ったため一次判断に使わないこと

## 完了済み（今セッション）

- 4 モード条件監査（軸 A: 記事明記 vs 仮定、軸 B: データ出所）→ DLS-010 + 原本
- 公式デフォルトの復元（`cosmos_framework/inference/defaults/*/sample_args.json`）と
  guidance 不一致の発見（ユーザー洞察「公式スクリプトを見れば分かるのでは」が起点）
- golden 照合（全 11 run FAIL）→ DLS-011 + 原本、検算（スケール一致・相関 0.808・
  ベースライン比較）と時間シフト構造の発見
- 原因の消去法: 最適化副作用 / fp16 / NATTEN / フレーム選択 / 初期ノイズ（arch_invariant_rand
  = CPU NumPy でアーキ不変）/ チェックポイント版差（3 snapshot の重み blob 同一）/
  公開コード drift / vision tower（policy 不使用）→ すべて棄却
- fp32 感度実験 E1〜E3 実装・実行・判定 → DLS-012 + 原本（数値精度仮説の全棄却）
- `scripts/check_policy_golden_mse.py` 追加（golden 自動 DL + 全 run 採点、stdlib のみ）
- コミット `7215021`

## 次のアクション

→ `tasks/todo.md` の `Active` セクションを参照（DLS-123: タスク本体は todo.md に一元化）

## ブロッカー・注意事項

- **未 push**: main が origin/main より 9 コミット先行
- **CUDA 参照 run はユーザー判断待ち**（環境調達が必要。todo.md Active 最上位）。
  これが決まるまで README の「同一条件」表現の訂正は保留が安全（原因帰属が未確定のため）
- **TeaCache 実験（DLS-004）は速度数値を報告しない**（DLS-003 制約）。9 run 中 2 run 完了、
  `calib_logonly` は `thresh_0.00` と同一設定で冗長（本セッション前半の分析、DLS 未起票）
- E1〜E3 のフラグは切り分け専用。**速度測定に使わない**（ヘルプにも明記済み）
- 記事側 guidance の確定には T2V `--guidance 6.0` 1 run（transformer_forward 2 倍の実証）が残っている
- Bash 出力が空になる事象が散発。回避策: スクラッチパッドのファイルにリダイレクトして Read

## 関連ファイル

- `.claude/.dls/active.md`（DLS-001〜003, 005〜012。DLS-004 は experiment ブランチ側）
- `.claude/.dls/raw/20260727_doc_article_conditions_and_official_defaults_audit.md`（軸 A/B + guidance）
- `.claude/.dls/raw/20260727_doc_policy_golden_mse_verification.md`（照合方法と全 run 結果）
- `.claude/.dls/raw/20260727_doc_policy_golden_mse_root_cause_analysis.md`（消去法と時間シフト）
- `.claude/.dls/raw/20260727_doc_policy_golden_mse_precision_sweep.md`（E1〜E3）
- `scripts/check_policy_golden_mse.py`（golden 採点の再現手順の正）
- `scripts/run_cosmos_framework_policy_rocm.py`（E1〜E3 フラグ）
- `result/attn_fp32_e1_20260727/` / `result/vae_fp32_e2_20260727/` / `result/model_fp32_e3_20260727/`
