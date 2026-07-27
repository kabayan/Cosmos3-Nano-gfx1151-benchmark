# 原本: tokenizer pin 法医学調査 — pin 無罪の確定と v1→v2 checkpoint 差し替えの発見

- 日付: 2026-07-27
- 種別: doc（調査ログ、`/dls-discuss 精度深堀り` から派生）
- 文脈: `/dls-discuss「精度に関して深堀 元実装で精度は出ている？」` の議論中、ユーザー指摘
  「記事によれば元のリポジトリとの精度を比較しているように見える」を受け、DLS-012 で
  最有力候補に昇格していた tokenizer pin 代替仮説の法医学的調査をユーザーが指示。
  すべてローカル照合 + HF/GitHub API のみで実施（GPU 実行なし）。

## 1. 議論の確定事項（調査前）

「元実装で精度は出ているか」は 3 つに分解される:

| 解釈 | golden action MSE < 0.05 | 状態 |
|---|---|---|
| (a) 最適化前の本環境コード（ROCm） | 不合格 0.126471 | 実測確定（DLS-011） |
| (b) 記事の環境（DGX Spark GB10・検証版） | 合格 0.013194 | 第三者による golden 照合の実測記録 |
| (c) 現公開コード b3967db + CUDA | 不明 | 未検証（DLS-012 の未決点） |

(b) は「記事の自己申告」ではなく、記事著者が元リポジトリの公式 golden・公式メトリクス
（`golden_mse_max 0.05` への言及で確証、DLS-010 監査）で照合した第三者実測である。

## 2. pin の正体と 404 の理由（確定）

- pin の場所: `cosmos_framework/inference/configs/model/Cosmos3-Nano.yaml` L182-185
  `vlm_config.tokenizer`（`build_processor_lazy`、repository `nvidia/Cosmos3-Nano`）
  → **VLM テキスト processor の revision pin**（video VAE ではない）
- 初回公開 commit `1bd5fdc36`（2026-05-31 "Import initial codebase"）:
  `revision: a18b727665f0dd03bc032229a6acb47ba11dc4cb`
- 翌日 commit `411d25b2e`（2026-06-01 "Apply origin/main diff: pin HF revisions to main (#7)"）:
  `revision: main` に変更
- **404 の理由**: HF `nvidia/Cosmos3-Nano` の main 履歴は 2026-06-01 の
  "Super-squash branch 'main' using huggingface_hub"（現 main の初 commit `03c14e74a`）で
  潰されており、a18b727 は squash で消された pre-squash commit。
- pre-squash 履歴は branch `spectralflight/shim`（tip `35c5cd345`、2026-03-10〜05-13 の
  16 commits）に残存。a18b727 はこの中にも無い = **5/13〜5/31 の squash 窓の commit**。

## 3. tokenizer 無罪の証拠（rejected_hypothesis の根拠）

- processor が読む全ファイル（tokenizer.json / tokenizer_config.json / vocab.json /
  merges.txt / chat_template.json / preprocessor_config.json /
  video_preprocessor_config.json）は **shim(5/13) と現行 main(411f42a8, 7/9) で
  git oid 完全一致**（HF tree API 照合）
- ローカル HF キャッシュ 3 snapshot（138d071c=6/1 期, 03c14e74=6/1 squash,
  411f42a8=7/9）でも同ファイル群は blob（symlink 先）完全一致
- a18b727（5/13〜5/31）だけ内容を変えて 6/1 までに戻した「二重反転」がない限り、
  tokenizer 内容は全観測期間で不変。**代替 revision 使用は golden MSE 不合格の原因ではない**

## 4. v1→v2 checkpoint 差し替えの発見（新事実・最重要）

- shim(5/13) の `checkpoint.json`: experiment **`cosmos3_ga_16bm8b_v1_midtrain`** /
  `iter_000012000` / `use_ema_weights: true`（現行 main の checkpoint.json は `{}`）
- 公開 framework の yaml `_metadata.args.experiment` は**初日 5/31 から一貫して
  `cosmos3_ga_16bm8b_v2_midtrain`**（1bd5fdc36〜67a53a116 まで全 commit で確認）
- transformer/（diffusers 形式）の全 7 shard が shim vs main で LFS oid 不一致。
  tensor 名は全面リネーム（814 key 中共通 2 のみ。`action2llm`→`action_proj_in`、
  `llm2sound`→`audio_proj_out`、`model.layers.*`→`layers.*` 等）
- **バイト照合**（HTTP Range で safetensors header を解析し対応 tensor を直接取得）:
  - `model.layers.0.input_layernorm.weight` [4096] BF16: corr **0.901646**、
    mean_abs_diff 4.5e-4（std 1.3e-3 比）、**95.95% の要素が相違**
  - `model.norm.weight` [4096]: corr 0.999998、0.44% のみ相違
  - → 再エクスポート/リネームではなく**別 checkpoint（同系統の継続学習像）**
- トップレベル `model.safetensors.index.json`: shim は旧 key 形式 1165 key / 8 shard、
  ローカル 3 snapshot はすべて新 key 形式 1165 key（= **本環境の全 run は v2 重み**）。
  key 数が同一なのでリネーム対応表は機械的に導出可能な見込み
- config.json も v1→v2 で形式移行 + vlm_config の意味論的フィールド差
  （qk_norm_for_text/qk_norm_for_diffusion の再編、tie_word_embeddings、
  layer_module、model_type qwen3_vl→cosmos3_omni）。ただし inference は framework 側
  yaml を読む（`load_model_config_dict` → `config_file_type: module`）ため
  config.json 自体は実行時に読まれない

## 5. golden の正体訂正（DLS-011/012 の前提修正）

`/tmp/cosmos-framework/inputs/omni/action_policy_robot.json`:

- **`golden_action_path` と `action_path` が同一ファイル**:
  `cosmos-dependencies` commit `2b17a2413bd8` の `inputs/action/bridge_20260501_0.json`
- 中身は 16×10 の action 配列（実取得で確認）= **データセット実測軌道（ground truth）**
  の可能性が高く、「golden は 2026-05-01 の内部コードで生成」という従来の枠組みは
  ファイル名の日付（episode 資産日付）との混同だった可能性が高い
- pin は commit 固定で、cosmos-framework 側の当該 JSON も公開以降変更なし
  （git 履歴は 1bd5fdc36 のみ）。**golden 側のドリフトは構造的に不可能。
  動いたのはモデル側（v1→v2）だけ**

## 6. 記事の公開日（確定）と時系列

記事「DGX Spark で NVIDIA Cosmos 3 を動かしてみた」（DevelopersIO）の公開日は
**2026-06-01**（記事ヘッダの time 要素で確認）。

| 日付 | 出来事 |
|---|---|
| 2026-05-01 | golden 資産の episode 日付（bridge_20260501、ground truth） |
| 2026-05-13 | HF main（pre-squash）= v1_midtrain iter12000 EMA 重み |
| 5/13〜5/31 | squash 窓: **checkpoint v1→v2 差し替え** + 全 tensor リネーム。a18b727 もここ |
| 2026-05-31 | framework 公開（1bd5fdc36）、a18b727 を pin、yaml は v2 前提 |
| **2026-06-01** | **記事公開（0.013194 PASS）** / HF super-squash / framework pin→main |
| 6/1〜7/9 | HF 重み・tokenizer とも不変（ローカル blob + tree API で実測） |

記事の検証実施は 6/1 以前 = v1 時代に跨がる可能性がある。もし記事が v1 重みで
検証していたなら、「公開 v2 checkpoint は golden 基準を誰も検証していない」
（CI は numeric golden 対象外、RTX 5090 zenn 記事は policy 未検証）という筋書きが
本環境の FAIL を含む全観測と整合する。

## 7. E4 実験提案（未実施・ユーザー判断待ち）

**v1 重み（shim revision `35c5cd345`、約 30GB）を取得し、本環境（ROCm）で
golden 照合を 1 run。**

- PASS（≈0.013）なら: ROCm スタック完全無罪 + 原因は v1→v2 差し替えと確定。
  CUDA 参照 run 不要
- 反対視点（リスク非対称性): v1 は旧 tensor 名 + 旧 config で公開コードに直接
  載らない。key リネーム + config 整合の移植が必要で、**FAIL はロード誤りと
  真の不一致を識別できない（PASS のみが情報を持つ）**
- 撤退ライン: リネーム移植の工数が想定を超えたら中断し CUDA 参照 run に切替

## 8. 検証再現手段（主要 URL / コマンド）

- HF refs/commits: `https://huggingface.co/api/models/nvidia/Cosmos3-Nano/refs`、
  `/commits/main`、`/commits/spectralflight%2Fshim`
- tree（oid 照合）: `/api/models/nvidia/Cosmos3-Nano/tree/<rev>?recursive=false`
- yaml 履歴: `gh api "repos/NVIDIA/cosmos-framework/commits?path=cosmos_framework/inference/configs/model/Cosmos3-Nano.yaml"`
- pin 初出: `https://raw.githubusercontent.com/NVIDIA/cosmos-framework/1bd5fdc36/...Cosmos3-Nano.yaml`（L185）
- tensor バイト照合スクリプト: セッション scratchpad `cmp_tensor.py`（safetensors header を
  HTTP Range で解析し data_offsets の範囲を直接取得、bf16→f32 変換して比較）
