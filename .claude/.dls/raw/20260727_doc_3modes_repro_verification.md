# T2I / T2V / I2V 公表値の再現検証（原本）

- 日時: 2026-07-26 23:10 〜 2026-07-27 00:31
- 検証者: CC
- 目的: README の公表値 T2I 27.136 秒 / T2V 32.165 秒 / I2V 25.045 秒 が現環境で再現するか確認する
  （DLS-006 で Policy の対外比較を訂正した際、他 3 モードは未検証のまま残っていた）
- 結論: **3 モードとも ±0.6% で再現。公表値は有効で訂正不要**。
  ただし再現には **TunableOp 表の読み込みが必須**であることが判明した。

## 1. 記録側の測定条件（今回特定したもの）

| 要素 | 値 | 出典 |
|---|---|---|
| Docker イメージ | `cosmos3-rocm72-diffusers:local` | docs/cosmos3-rocm72-diffusers-image.md |
| diffusers | イメージ同梱 0.39.0.dev0（`/opt/diffusers`、und branch cache パッチ入り） | 同上 / docs/cosmos3-rocm-i2v-und-branch-cache-diffusers-native.md |
| pip install | **実行しない**（`COSMOS3_DIFFUSERS_INSTALL=true`） | 同上 |
| variant | `aotriton_tuned` = AOTriton experimental + **TunableOp 表** | scripts/run_rocm_speed_matrix.py `VARIANTS` |
| TunableOp 表 | `result/rocm_speed_matrix/tunableop_results0.csv`（Jun 3、md5 `0e28785495b0b8d1e002ce7ed759337b`） | 実物 |

`aotriton_tuned` の完全な env:

```
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_RECORD_UNTUNED=0
PYTORCH_TUNABLEOP_FILENAME=/workspace/result/rocm_speed_matrix/tunableop_results%d.csv
```

T2I の記録 run は runner の case ではなく手打ちで、コマンドは
`docs/cosmos3-rocm-t2i-non-gemm-improvement-plan.md` L289-306 に残っていた。
T2V / I2V は `case_command("t2v_article_warm_full" / "i2v_article_und_cache_warm_full", "aotriton_tuned")`
が生成する文字列と out-dir 以外一致することを確認済み。

## 2. 3 回の試行

| run | イメージ | diffusers | env | 出力先 |
|---|---|---|---|---|
| v1 | 既定 `rocm/pytorch:rocm7.2_...` | git main（7 月） | AOTriton のみ | `result/verify_3modes_20260726/` |
| v2 | 記録イメージ | 同梱パッチ版 | AOTriton のみ | `result/verify_3modes_v2_20260726/` |
| **v3** | 記録イメージ | 同梱パッチ版 | **aotriton_tuned 全体** | `result/verify_3modes_v3_20260726/` |

v1 の顛末:
- T2I: `RuntimeError: Transformer does not expose enable_und_branch_cache`。
  7 月の diffusers main にパッチ API が無いため。スクリプト L390-392 は hard fail する
- T2V: コンテナ内 pip が DNS 失敗（`Could not resolve host: github.com`）。一過性
- I2V: 完走。`UndBranchCachePrototype`（monkeypatch）へフォールバックして 31.877 秒

## 3. 結果（measured run）

| mode | 指標 | 記録 | v3 | v3 差 | v2 | v2 差 |
|---|---|---|---|---|---|---|
| T2I | 総時間 | 27.136 | **27.214** | +0.3% | 32.641 | +20.3% |
| T2I | transformer_forward | 24.837 | 24.938 | +0.4% | 30.349 | +22.2% |
| T2I | vae_decode | 1.746 | 1.730 | −0.9% | 1.744 | −0.1% |
| T2I | unattributed | 0.545 | 0.538 | −1.3% | 0.540 | −0.9% |
| T2I | vae warmup | 659.752 | 659.679 | −0.0% | 659.981 | +0.0% |
| T2V | 総時間 | 32.165 | **32.156** | −0.0% | 46.765 | +45.4% |
| T2V | transformer_forward | 26.794 | 26.836 | +0.2% | 41.501 | +54.9% |
| T2V | vae_decode | 4.151 | 4.111 | −1.0% | 4.059 | −2.2% |
| T2V | vae warmup | 369.591 | 375.275 | +1.5% | 373.251 | +1.0% |
| I2V | 総時間 | 25.045 | **24.905** | −0.6% | 31.837 | +27.1% |
| I2V | transformer_forward | 19.680 | 19.560 | −0.6% | 26.509 | +34.7% |
| I2V | vae_decode | 4.134 | 4.115 | −0.5% | 4.098 | −0.9% |
| I2V | vae warmup | 370.074 | 373.513 | +0.9% | 372.590 | +0.7% |

und branch cache 統計は記録と v3 で完全一致:
- T2I: `transformer_calls 70 / write 1 / read 69 / invalidations 1 / cached_layers 36 / cache_gib 0.882`
- I2V: 同上、`cache_gib 0.784`
- T2V: 未使用（記録は当時のスクリプトがフィールドを出力せず `{}`、v3 は `enabled: false`）

TunableOp 表の md5 は実行後も `0e28785495b0b8d1e002ce7ed759337b` のまま（`TUNING=0` / `RECORD_UNTUNED=0`
のため書き戻しは起きない）。記録アーティファクトは無傷。

## 4. 診断の経緯（誤報を出しかけた記録）

v2 の時点で「+20〜55% の退行」に見えた。ただし次の 2 点が退行仮説と整合しなかった:

1. `vae_decode` が 3 モードとも記録と一致（−0.5〜−2.2%）。VAE warmup も T2I で −0.0%。
   環境全体が劣化したのなら conv3d 経路も遅くなるはず
2. 乖離が `transformer_forward` にのみ現れた。GEMM 主体の経路だけが遅い形

この形は「GEMM の調律表が効いていない」場合の予測と一致するため variant 定義を確認し、
`aotriton_tuned` が TunableOp 3 変数を含むことを発見した。

紛らわしかった点: summary の `tunableop_config.enabled` は **CLI の調律パラメータ指定のみ**を反映し、
環境変数 `PYTORCH_TUNABLEOP_ENABLED` を反映しない（`configure_tunableop` L591-597。
`max_tuning_duration` 等が全て None なら無条件に `{"enabled": False}` を返す）。
そのため記録側の summary も `enabled: false` と表示されており、
「記録も TunableOp 無しで測った」と誤読しうる状態だった。

## 5. Policy との非対称性

DLS-005 は Policy について「TunableOp 表の適用を再現条件に含める」を dormant にしている
（v3 run が TunableOp 無しで 42.88 秒 = 記録比 +2.9% に到達したため）。
この判断は **Policy 経路に限って正しい**。diffusers 経路（T2I/T2V/I2V）では表が必須で、
無しでは transformer_forward が +22〜55% 遅くなる。

追加の非対称性: DLS-005 は `PYTORCH_TUNABLEOP_TUNING=0` での表読み込みが Policy で
`Expected iter != ops_.end()` によりクラッシュすると記録している。diffusers 経路では
同じ設定が正常に動作する（記録 run も v3 も成功）。

表の寄与を transformer_forward で見ると T2I +22% / T2V +55% / I2V +35%。
`docs/cosmos3-rocm-t2i-non-gemm-improvement-plan.md` の「GEMM 表は約 1% 改善」は
Stream-K 対応版が基底表に**上乗せ**する分の話であり、基底表そのものの寄与とは別物。

## 6. 副次的に確定したこと

**diffusers の版差は 3 モードの速度に影響しない**。
I2V で比較すると、
- v1（7 月 git main + prototype monkeypatch キャッシュ）transformer_forward 26.592 秒
- v2（6 月パッチ版 + native キャッシュ）transformer_forward 26.509 秒

差 0.3%。und キャッシュ統計も両者・記録の 3 つで完全一致。
prototype 実装と native 実装の等価性も、この一致が実測で裏付けている。

## 7. 留保

- 各モード measured run は 1 本ずつ。ノイズフロアは未測定。
  ただし 3 モードが独立に ±0.6% で一致し、モードごとに 5〜6 指標が同時に一致しているため、
  偶然の一致とは考えにくい
- 本検証は「公表値が再現するか」のみを見た。und branch cache 由来の
  楽観バイアス（T2I / I2V に約 1〜1.5 秒と前セッションで見積もり）が
  対外比較として妥当かは別問題で、本検証では判断していない
