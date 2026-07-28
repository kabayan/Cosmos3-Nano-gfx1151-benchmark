# 原本: E4 — v1_midtrain 重みによる ROCm golden 照合

- 日付: 2026-07-28
- 種別: doc（実験ログ）
- 文脈: DLS-013 で定義した決着実験 E4 をユーザー承認のうえ実行する。
  仮説は「golden MSE 不合格（DLS-011）の原因は公開前 squash 窓での
  checkpoint v1_midtrain → v2_midtrain 差し替えであり、v1 重みなら合格する」。
  PASS のみが情報を持つ非対称設計（FAIL はロード誤りと識別できない）。

## 1. 移植方式の訂正 — 手動リネームは不要だった

DLS-013 / todo.md では「旧 814 key を新形式へ機械リネームして移植する」と
想定していた。**これは不要**と実測で確定した。

公開コード側 `cosmos_framework/inference/model.py` の
`_DIFFUSERS_KEY_MAPPING_RES`（L72-）は diffusers 形式 checkpoint の key を
`OmniMoTModel.net` の内部 key へ写す正規表現テーブルで、**旧形式・新形式の
両方の綴りを同時に受理する**。具体的には

- `^action_proj_in\.` → `action2llm.`、`^audio_proj_out\.` → `llm2sound.`
- `\.self_attn\.to_q\.` → `.self_attn.q_proj.`、`add_q_proj` → `q_proj_moe_gen` 等
- `^model\.(embed_tokens\.|layers\.|norm\.)` と `^(embed_tokens\.|layers\.|norm\.)`
  の両方を `language_model.model.` へ写す規則

が並んでおり、v1 の綴り（`model.layers.*` / `action2llm.*`）は前段の規則に
当たらず後段でそのまま正しい net key に落ちる。

**実証**（両 index.json に対して同テーブルを適用したシミュレーション、
scratchpad `mapsim.txt`）:

| | v1 (shim 35c5cd345) | v2 (main 411f42a8) |
|---|---|---|
| index の key 数 | 1165 | 1165 |
| net key へ写った数 | 1165（drop 0、衝突 0） | 1165（drop 0、衝突 0） |
| 写像後の net key 集合 | **完全一致** | |

したがって v1 の shard はリネームせずそのまま公開ローダに載る。
E4 の主要コストと想定していた「リネーム移植」は消滅し、撤退ライン
（工数超過なら CUDA run に切替）は発動しない。

## 2. アーキテクチャは v1/v2 で不変 — 動くのは重みの値だけ

`transformer/config.json` を v1 と v2 で全キー比較した結果、差分は
`_diffusers_version` の `0.37.0` → `0.37.1` の**1 行のみ**。
hidden_size / head_dim / intermediate_size 等の構造パラメータは全一致。

これは E4 の帰属を強める。v1/v2 で構造が違えば「構造差か重み差か」を
切り分けられないが、構造が同一である以上、照合結果の差は重みの値にのみ
帰属する。あわせて DLS-013 で観測した layer0 layernorm の 96% 相違が
「別 checkpoint」であることの傍証にもなる（同一構造・別値）。

なお config.json（ルート）は inference 時に読まれない（framework 側 yaml を
`--config-file` で与えるため）。v1/v2 の config.json 形式差は E4 に影響しない。

## 3. v1 checkpoint 側の実務的な瑕疵

shim revision のルート `model.safetensors.index.json` の weight_map は、
vision_encoder の 351 key について**エクスポート環境の絶対パス**
`/tmp/root/Cosmos3-Nano/vision_encoder/model.safetensors` を指している
（v2 では相対パス `vision_encoder/model.safetensors`）。

公開ローダの `_should_drop_diffusers_weight_path` は
`^(?!transformer/|vision_encoder/)` に一致するパスを落とすため、この絶対パスは
**vision タワーの重み 351 個が丸ごと無言で捨てられる**方向に働く。
取得後に相対パスへ書き換えて回避した。

（policy 推論は vision 理解タワーを使わない（DLS-012 で確認済）ため出力への
影響は無いと考えられるが、ローダの missing_keys 検査を通すために修正が要る。）

## 4. 実行環境 — コンテナ手順を repo に記録した

cosmos-framework 経路の実行コンテナの作り方は repo のどこにも記録が無く、
今回も再発見が必要だった。これは DLS-005（headline 数値の再現手順が未記録で
約 2 時間の誤報調査を招いた）と同型の欠落であり、同じ再発を防ぐために
成果物として残す:

- `docker/cosmos3-rocm72-framework.Dockerfile` — ROCm ベースイメージ + framework 依存
- `scripts/run_cosmos_framework_policy_docker.sh` — マウント・環境変数込みの起動ラッパー

依存パッケージは pyproject の `[project].dependencies` だけでは不足し、
実測で `iopath` / `multi-storage-client==0.44.0` / `boto3` / `wandb` /
`qwen_vl_utils` の追加が必要だった（import 到達順に反復して確定）。
diffusers は git main だと `huggingface-hub>=1.23` を要求して
transformers 4.57 系（`huggingface-hub<1.0`）と解決不能になるため、
リリース版 diffusers を使う。

## 5. 実行条件

```
scripts/run_cosmos_framework_policy_docker.sh \
  --policy-checkpoint-path /v1ckpt \
  --policy-config-file /workspace/tmp/cosmos-framework/cosmos_framework/inference/configs/model/Cosmos3-Nano.yaml \
  --policy-model-size 8B \
  --out-dir /workspace/result/v1_ckpt_e4_20260728 --warmup-runs 0
```

- 重み以外は公開レジストリと同一条件にするため、config は checkpoint 同梱の
  config.json ではなく framework の `Cosmos3-Nano.yaml` を明示指定する
- tokenizer / VAE / sound tokenizer は HF main（= v2 側）から取る。tokenizer は
  DLS-013 で全観測期間不変と実証済みのため変数にならない
- 入力・seed は既存 run と同一（`inputs/omni/action_policy_robot.json`）

## 6. 結果

（実行後に追記）

---

（上の §6 プレースホルダに対する実行後の追記。原本は追記のみのため以下に記す）

## 6'. 結果 — FAIL（0.248372）、しかも v2 より悪化

`result/v1_ckpt_e4_20260728/`（2026-07-28 01:00 開始 / 01:44 完了、サンプリング 1.15 s/it）

| run | 重み | golden MSE | 判定 |
|---|---|---:|---|
| **v1_ckpt_e4_20260728** | **v1_midtrain iter12000 EMA** | **0.248372** | **FAIL** |
| classmethod_policy_framework | v2_midtrain（最適化前） | 0.126471 | FAIL |
| mainline_full_v4_20260726 | v2_midtrain | 0.128000 | FAIL |
| 記事（DGX Spark） | 不明 | 0.013194 | PASS |

v1 は v2 の約 2 倍悪い。次元別でも dim2 / dim3 を除く全次元で v1 が劣り、
グリッパー dim9 は 1.017（v2 は 0.507）。

### 6'.1 ロードは正しく行われた（FAIL の非識別性の一部を潰した）

事前登録どおり FAIL 自体はロード誤りと真の不一致を識別しないが、以下により
「重みが読まれなかった / 無視された」線は消えた。

- 事前スモーク: 1165 key すべてが net key へ写り（drop 0 / 衝突 0）、
  shape・dtype が v2 と完全一致
- `_DiffusersLoadPlanner` は missing_keys があれば例外を投げるが、run ログに
  ロード関連の warning / error 無し
- **出力が実際に動いた**: `MSE(v1_run, v2_baseline) = 0.149132` に対し、
  v2 同士の run 間ノイズは `0.001301`。100 倍以上離れており重みは確かに効いている

### 6'.2 DLS-013 の「別 checkpoint」判定は過大だった（要訂正）

全 1165 tensor をローカル実体で比較した（scratchpad `diff_all.py`）。

| 指標 | 値 |
|---|---:|
| bitwise 一致した tensor | 43 / 1165 |
| 相対平均絶対差の中央値 | **0.0128** |
| 同 平均 / 最大 | 0.0430 / 0.3990 |

差は**生成（moe_gen / diffusion expert）経路に集中**している。上位は
`action2llm.bias`（0.399）、`time_embedder.mlp.2`（0.388）、
`action_modality_embed`（0.335）、`layers.*.{self_attn,mlp}_moe_gen.*`（0.24〜0.31）、
`vae2llm`（0.260）。一方で理解（und）経路 — `q_norm` / `k_norm` / `lm_head` /
`embed_tokens` / `visual.blocks.*` — はほぼ同値で、一部は bitwise 一致。

DLS-013 が「別 checkpoint」の根拠にした `layers.0.input_layernorm.weight` は、
「96% の要素が相違」は再現した（0.9595）が、**相対差は 4.2% にすぎない**。
corr 0.90 から「別系統の checkpoint」と読んだ解釈は過大で、実態は
**同一系統の継続学習（生成側を中心に更新）**。v1 → v2 は作り直しではない。

### 6'.3 「記事は v1 で合格した」説は支持されない

記事が v1 重みで 0.013194 を出していたのなら、同じ v1 重みを使う本 run も
0.013 付近に来るはずで、実際は 0.248 と 19 倍悪い。したがって DLS-013 の
assumption の主線（公開前 v1→v2 差し替えが golden 不合格の原因）は**弱まった**。
むしろ golden に対しては v2 の方が v1 より近い。

残る交絡: v1 の `config.json` は現行 yaml と意味論差を持つ
（`qk_norm_for_text` v1=False / 実行時=True、`qk_norm_for_diffusion`、
`tie_word_embeddings` v1=True / 実行時=False、`layer_module`
v1=`Qwen2MoTDecoderLayer` / 実行時=None）。`unified_mot.py` L470 では
`qk_norm_for_text` が False だと q_norm/k_norm が `nn.Identity` になるため、
設定不一致は原理的に出力を壊しうる。

ただしこの交絡は当初懸念より小さい可能性が高い。v1 の
`self_attn.q_norm.weight` は all-ones ではなく学習済みの値を持ち、しかも
v2 の対応 tensor とほぼ同値（mean 1.743011 vs 1.742882、max とも 3.453125）。
und 側 QK norm は v1 でも実在していたと考えられ、shim の config.json の
`qk_norm_for_text: False` はエクスポート時のメタデータであって v1 学習時の
実構成を反映していない可能性がある。

### 6'.4 帰結

- E4 は仮説を確証しなかった。「重みを差し替えれば通る」という単純版は棄却
- golden 不合格の原因は v1→v2 の checkpoint 差し替えでは説明できない
- 決着手段は当初の CUDA 参照 run に戻る（公開コード + 公開 v2 重み + CUDA で
  0.013 が出るか）。出れば本環境固有、出なければ公開資産では記事値は再現しない

## 7. 付随: カーネルキャッシュ永続化（コールドスタート短縮）

コンテナが `--rm` のため MIOpen のカーネル探索結果・コンパイル済みバイナリと
inductor / triton のコンパイル結果が毎 run 捨てられていた（E4 の run ログにも
`SingleProcess AUTOTUNE benchmarking ... for 9 choices` が多数）。ホストへ逃がして
持ち越すようにした（`scripts/run_cosmos_framework_policy_docker.sh`、
`CACHE_DIRS=0` で無効化可）。

実測（`scripts/probe_kernel_cache_persistence.py`、conv3d 2 形状、`result/kcache/`）:

| 条件 | total | 大 conv の初回呼び出し |
|---|---:|---:|
| マウント無し | 15.75 s | 10.76 s |
| マウント有り 1 回目 | 15.55 s | 10.58 s |
| マウント有り 2 回目 | **13.48 s** | **8.52 s** |

持ち越しでプローブ全体 −14%、大 conv の初回 −21%。ホストの
`~/.cache/miopen/3.5.1.5b515cf1bc/gfx1151_20.ukdb` は 188KB → 270KB に増えており、
コンテナからの書き込みが実際に永続化されている。

本 run（Policy）はこのプローブより遥かに多い形状と inductor autotune を通るため
効果は上振れする見込みだが、**未実測**。次の Policy run で総所要を比較する。

留保: これらは実行方法のキャッシュであって計算内容を変えない（DLS-003 の
計算省略系ではない）が、find-db の有無で MIOpen のアルゴリズム選択が変わりうるため
bit 一致は保証しない。DLS-012 の fp32 感度実験が示す非感受性の範囲に収まる想定で、
次の Policy run の golden MSE が既存帯（0.126〜0.134）に入ることで確認する。
