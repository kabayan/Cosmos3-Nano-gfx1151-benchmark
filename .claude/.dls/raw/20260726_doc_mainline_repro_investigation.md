# 原本: 本線ベンチ 41.66 秒の再現性調査（退行アラームの誤報とその原因）

- 日付: 2026-07-26
- 種別: doc（調査ログ）
- 文脈: TeaCache 品質評価（DLS-004）の実験中、CC が「本線が 7.7 倍遅くなっている」と報告。
  ユーザー指示「比較すべきは元記事」「退行の根本原因調査を優先」により品質評価を中断して調査した。

---

## 発端（誤った観測）

TeaCache 品質評価 run（warmup なし単発）の所要時間を見た CC が、旧ベンチのコールド 1 回目
（969.85s）と今回のコールド 1 回目（1867.3s）を比較して「約 2 倍の劣化」と報告した。
ユーザーが「この比較は無意味、比較すべきは元記事」と指摘。

これを受けて本線プロトコル（warmup 2 回 + 測定 run）を再実行したところ
`generate_batch = 322.21s` となり、記録 41.66s に対し **7.7 倍**の乖離が出た。
CC はこれを「本線退行の可能性」と報告した。→ **この報告自体が誤りだった**（後述）。

## 調査で確定した事実

1. **コード無変更**: `scripts/run_cosmos_framework_policy_rocm.py` は 6/14 コミット `9e39392` 以降、
   TeaCache コミット `3d3b514` で 266 行の純粋追加のみ（削除 0 行）。本線経路は当時と同一。
   DLS-004 の「既定 OFF で本線経路は完全無変更」は裏付けられた。
2. **フレームワーク無変更**: `/tmp/cosmos-framework`（temp_src の rsync コピー）は 6/13 付のまま。
3. **Docker イメージ無変更**: `rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1` の
   Created は 2026-01-22。6 月時点から同一。
4. **GPU クロック正常**: amd-smi 実測で SCLK 2899MHz（レベル 1/2、上限 2900MHz）、
   MCLK 1000MHz（**このAPUの最上位レベル**。400/800/1000 の 3 段構成）。
   rocm-smi の "low-power state" 警告と MCLK 1000MHz 表示から疑った ROCm #5750
   （Strix Halo 低クロック張り付き）は**非該当**。
5. **TunableOp 表は現行スタックで直接は使えない**: `tunableop_results00.csv`
   （6/14 18:12 作成、記録 run 18:32 の 20 分前）のバリデータ
   （PT 2.9.1 / HIP 702 / gfx1151 / hipBLASLt 100201-5b515cf1bc）は現行イメージと一致するが、
   `PYTORCH_TUNABLEOP_TUNING=0` で読み込むと VAE encoder の SDPA 到達時に
   `RuntimeError: Expected iter != ops_.end()` でクラッシュした。
   最終報告 §④ の記載は `TUNING=1`（オンザフライ調律）であり、CC の `TUNING=0` 指定が誤り。

## 根本原因

**記録 41.66s は `--policy-condition-cache` を付けて測定されており、CC の再現 run はそれを
渡していなかった。** 同フラグは 2 つの効果を持つ（script L234, L258, L309-318）:

1. warmup 時に計算した conditioning を保存し、measured フェーズでは再計算せずキャッシュを返す
2. プロファイラ経路を有効化し、ステージ境界で同期を取る（= 「同期総和」測定が成立する）

フラグ無しでは (1) により conditioning 計算が測定 run に丸ごと乗り、(2) が無いため decode が
非同期のまま 0.02s と記録され実処理が計測外へ流出する（実際にこの症状が出た）。

## 3 者比較（決定的証拠）

| run | generate_batch | generate_samples | decode | その他 |
|---|---|---|---|---|
| 記録 2026-06-14 | **41.66s** | 33.84 | 7.49 | 0.33 |
| v1 再現（フラグなし） | 322.21s | 132.35 | 0.02 | 189.84 |
| **v3（`--policy-condition-cache` のみ、TunableOp 無し）** | **42.88s** | 35.25 | 0.01 | 7.62 |

`--policy-condition-cache` を付けるだけで記録比 **+2.9%** に着地。**環境退行は存在しない**。
TunableOp は 41.66s 到達に不要だった（v3 は未使用で 42.88s）。

decode の計上先がフェーズで変わる（v3 warmup2 では decode=7.4985s と記録の 7.49s にほぼ一致、
measured では 0.01s で同額が「その他」に出る）が、比較すべき総和 `generate_batch` は一致する。

## 派生して判明した論点（未決）

`--policy-condition-cache` は measured フェーズの conditioning 計算を warmup の結果で置換する。
v3 の実測で conditioning は約 81s 相当（warmup2 の generate_samples 116.70s − measured 35.25s）。
つまり **41.66s / 42.88s は conditioning 約 81s を測定対象から除いた数値**である。

元記事（DGX Spark 21s / 論文値 29.00s）が conditioning を含むか不明のため、
「同一条件比較」の妥当性は現時点で判定できない。ユーザー判断を要する未決事項として残す。

## 再現コマンド（今後の正）

```bash
docker run ... -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 ... \
  python /workspace/scripts/run_cosmos_framework_policy_rocm.py \
    --out-dir <out> --warmup-runs 2 --policy-condition-cache
```

生成物: `result/mainline_repro_v3_20260726/`（42.88s）、
`result/mainline_repro_20260726/`（322.21s、フラグ無しの反例）。

## コスト

この誤報と調査で約 2 時間を消費し、TeaCache 品質評価（本来の主タスク）が中断した。
原因は「headline 数値 41.66s の再現手順が repo のどこにも記録されていなかった」こと。
最終報告 §④ は TunableOp に言及するが `--policy-condition-cache` には触れていない。
