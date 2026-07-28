# und branch cache 2 スロット化の実装と検証実測（DLS-017 採用案 A の執行）

- 日付: 2026-07-28
- 種別: doc（実装 + 検証実測の原本）
- 実施: CC（/dls-continue による todo.md Active 先頭実行可能タスクの自律実行）
- 前提判断: DLS-017（採用案 A = und branch cache の 2 スロット化、厳密キャッシュのまま read 化回復）

## 1. 実装内容

対象: `third_party/diffusers/src/diffusers/models/transformers/transformer_cosmos3.py`
（third_party/diffusers クローン内コミット `f829105c7`、親 `788b5f06f`）

- 単一署名スロット（`_und_branch_cache_signature` + layer→entry dict）を、**署名をキーとする
  LRU 2 スロット構造** `_und_branch_cache_slots: OrderedDict[signature, {layer: entry}]`
  （`_und_branch_cache_max_slots = 2`）に置き換え
- read 条件は「署名完全一致 かつ 全 36 layer 分のエントリが揃っている」— 厳密キャッシュのまま
  （近似なし）。CFG 下で交互に来る cond/uncond の 2 署名が互いを invalidate せず共存する
- スロット逸脱（第 3 の署名）は LRU 追い出し + `invalidations` カウント
- stats に `cached_slots` を追加。既存キー（transformer_calls / write_calls / read_calls /
  invalidations / cached_layers / cache_gib）は維持（cached_layers は全スロット合計に変更）
- 公開 API（enable/disable/reset/get_und_branch_cache_stats）のシグネチャ不変。
  ベンチスクリプト側の変更は不要

## 2. イメージ同期（/opt/diffusers）

- 方式: **docker cp + docker commit**（同一タグ `cosmos3-rocm72-diffusers:local` に再コミット）
  - 旧イメージ: `sha256:eab19ad6eb66580c931575167865c5a636766b10c46e9d85db70f19e1ffcd093`（ロールバック用に記録）
  - 新イメージ: `sha256:554e0573ec8911e6cf2b6ca977905ff1fd4b8509c7a8904dd22f314da7d32710`
- rebuild（`scripts/build_cosmos3_rocm72_diffusers_image.sh`）を採らなかった理由:
  Dockerfile は `COPY . /opt/diffusers` の後に pip install 層があるため、コード変更で
  pip 層が invalidate され再実行される。pip のネットワーク依存（DNS 失敗で run が落ちた
  DLS-007 の記録）を避け、変更ファイル 1 個の cp で完結させた
- 同期検証: コンテナ内 `diff -q` で third_party とイメージ内ファイルの一致を確認（SYNC_OK）

## 3. 検証実測プロトコル

- `result/guidance_2slot_20260728/run_commands.sh`（result ディレクトリに同梱）
- プロトコルは `result/guidance_official_20260728/run_commands.sh` と同一
  （DLS-007 v3 再現条件: イメージ + aotriton_tuned env + TunableOp 表読み込み、pip install なし。
  guidance は公式デフォルト T2I 4.0 / I2V 6.0、steps 35、--und-branch-cache）
- 対象は cache を使う T2I / I2V の 2 モード（T2V は cache 不使用のため対象外）
- 全 run exit 0

## 4. 結果

### (a) cache stats — 全スラッシュ解消

| モード | 2 スロット化前（guidance_official） | 2 スロット化後 |
|---|---|---|
| T2I | 140 calls / **140 writes / 0 reads** / 140 invalidations | 140 calls / **2 writes / 138 reads** / 0 invalidations / cached_slots 2 / cache 0.903 GiB |
| I2V | 140 calls / **140 writes / 0 reads** / 140 invalidations | 140 calls / **2 writes / 138 reads** / 0 invalidations / cached_slots 2 / cache 1.947 GiB |

writes 2 = cond/uncond 各 1 回（第 1 step のみ）。期待どおり。

### (b) 出力一致 — 厳密性の実証

| モード | 2 スロット化前 md5 | 2 スロット化後 md5 | 判定 |
|---|---|---|---|
| T2I jpg | `1b5c6bfd8b3a555923c3cf207791e916` | 同左（warmup / measured とも） | ビット一致 |
| I2V mp4 | `be7b15655679d7237b331147f4522800` | 同左 | ビット一致 |

決定性の土台: 前回 run 内で warmup/measured の jpg md5 が一致しており、
diffusers 経路は同条件でビット決定的。その上で修正前後も一致 = キャッシュは厳密。

### (c) 速度 — measured run（公式 guidance 条件）

| モード | 前 (s) | 後 (s) | transformer_forward 前→後 | 対記事倍率 前→後 | DLS-017 概算 |
|---|---:|---:|---|---|---|
| T2I | 115.589 | **49.633** | 113.254 → 47.340 (70 calls) | 5.25x → **2.25x** | ≈2.4x |
| I2V | 192.521 | **45.622** | 187.087 → 40.238 (70 calls) | 11.33x → **2.69x** | ≈2.6x |

- 対記事倍率の基準は DLS-016 と同一（T2I 記事 22.017 秒相当 / I2V 16.992 秒相当）
- DLS-017 assumption（cache read 経路 per-call の再現、confidence: medium)は
  実測で概ね確認。T2I は概算より良い（assumption 内の「uncond 側は系列長が短く
  概算より良くなる可能性」の方向）。I2V は概算どおり
- T2V は 2.53x のまま（cache 不使用、DLS-017 の C 案は dormant 継続）
- 価格差 2.0 との関係: 3 モードとも依然 2.0 超（T2I 2.25x / T2V 2.53x / I2V 2.69x）。
  DLS-017 の「CFG 条件での 2.0 到達は同一計算内容の制約下では非現実的」の確定見通しと整合

## 5. 生成物

- `result/guidance_2slot_20260728/`（run_commands.sh・driver.log・各 summary・出力 jpg/mp4・run.log）
- third_party/diffusers コミット `f829105c7`
- イメージ `cosmos3-rocm72-diffusers:local` = `sha256:554e0573ec89...`
