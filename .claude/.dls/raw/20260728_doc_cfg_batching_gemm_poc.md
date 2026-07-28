# 原本: CFG バッチ化 PoC（GEMM マイクロベンチ）— 候補 B の棄却

- 日付: 2026-07-28
- 種別: doc（実験ログ / PoC）
- 文脈: `/dls-plan CFG条件チューニング`。DLS-016 で「公式 guidance 条件では 3 モードとも
  価格差 2.0 超過」が確定し、2.0 到達には per-call 約 20% の追加短縮が必要と算出された。
  ユーザー選択は **「B を先に PoC」** — 構造的候補 B（CFG の cond/uncond を 1 回の
  forward にバッチ化し重み読み出しを償却する）の成立性を先に確かめ、
  結果を見てから A（und branch cache の 2 スロット化）の実装順序を決める。

## 1. B の成立条件と検証方法

CFG は現状 cond / uncond を**逐次 2 回** forward する（`pipeline_cosmos3_omni.py`
L1597-1667、DLS-016）。1 回にバッチ化して得があるのは、GEMM が**重み読み出し律速**
（memory-bound）の場合に限る。その場合、トークン数を 2 倍にしても重み読み出しは
1 回で済むため所要は 2 倍未満になる。逆に**演算律速**（compute-bound）なら
トークン 2 倍 = 所要 2 倍で、バッチ化しても得は無い。

判定指標: 同一重みに対しトークン数 N と 2N で GEMM を測り、**ratio = t(2N)/t(N)**。
- ratio << 2.0 → 重み読み律速。B に構造的余地あり
- ratio ≈ 2.0 → 演算律速。B に余地なし

形状は `result/rocm_speed_matrix/tunableop_results0.csv` の実測形状から採用
（Cosmos3 transformer の Linear: hidden 4096 / FFN 12288 / qkv 1024、
実際に現れるトークン数 N = 261 / 672 / 900 / 1904 / 2141）。

`scripts/probe_cfg_batching_gemm.py`（新規）。TunableOp は N と 2N で調律状態が
非対称になるため**無効化**（`PYTORCH_TUNABLEOP_ENABLED=0`）して土台を揃えた。
bf16、warmup 5 + 20 iter × 3 セットの最小値採用。

## 2. 結果

`result/cfg_batch_probe/gemm_bf16.json`

| layer (K→M) | N=261 | N=672 | N=900 | N=1904 | N=2141 |
|---|---:|---:|---:|---:|---:|
| qkv_proj (4096→1024) | 1.267 | 1.471 | 2.348 | 2.495 | 2.163 |
| attn_out (4096→4096) | 2.407 | 2.157 | 2.099 | 1.840 | 1.762 |
| ffn_up (4096→12288) | 1.144 | 1.993 | 2.101 | 1.695 | 1.631 |
| ffn_down (12288→4096) | 2.312 | 2.070 | 1.882 | 1.915 | 1.776 |

- **全体 ratio 平均 = 1.926**（min 1.144 / max 2.495）
- **N ≥ 672 に限ると ratio 平均 = 1.962**（実運用のトークン帯）
- 達成 TFLOPS は N と 2N でほぼ不変（例 ffn_up N=672: 31.42 → 31.53 TF、
  attn_out N=900: 30.31 → 28.88 TF）。一方 GB/s は N とともに単調低下
  （qkv N=261 の 103 GB/s → N=2141 の 42.6 GB/s）

**演算律速の署名**（TFLOPS 一定・GB/s 低下）が明確。トークン数を倍にすると
所要もほぼ倍になる。

## 3. 判定 — B は棄却

バッチ化の利得は 2.0 − 1.926 = **約 3.7%**、実運用帯（N≥672）では **約 1.9%**。
必要な per-call 短縮 20% に遠く届かない。

加えて実装上、CFG の cond / uncond は系列長が異なる（uncond はテキストのみで短い）
ため、バッチ化しても GEMM は「cond_len + uncond_len」トークンの 1 回になるだけで、
総トークン数は変わらない。演算律速である以上、総トークン数が同じなら所要も同じ。
**B は「2 回呼ぶ」ことが問題ではなく、計算量そのものが 2 倍になることが本質**
だった、というのが PoC の結論。

## 4. 副次的に判明したこと

観測された GEMM の最大スループットは **36.11 TFLOPS**（ffn_up N=1904 の 2N ケース）で、
README §4 の bench_peak BF16 実測値 **20.91 TFLOPS**（16384³）を上回る。
transformer の実形状は 16384³ 正方より効率が良い領域にある。

これは「GEMM 経路にはもう 20% の余地が無い」という B 棄却の傍証になる
（既にピーク近傍ないしそれ以上で回っている）。同時に、README §4 の
「ピークスループット」を性能上限の根拠に使う場合、形状依存で上振れすることの注記が要る。

## 5. 帰結 — 候補の再評価

B が消えたことで、dormant だった候補群を棄却事実を踏まえて再評価する（第 3 の道）:

- **A（und branch cache の 2 スロット化）**: B と独立に成立する。CFG 下で
  140 calls / 140 writes / **0 reads** の全スラッシュ（DLS-016）を、cond/uncond 各 1
  スロットで read 化に戻す。厳密キャッシュのまま（近似ではない）。
  概算回復: T2I 5.25x → ≈2.4x、I2V 11.33x → ≈2.6x
- **C（T2V への und cache 適用）**: T2V は現行 cache 未使用。ただし T2V の 2.53x は
  cache 無関係の純粋な計算量 2 倍であり、cache 適用で削れるのは und 枝の再計算分のみ
- **2.0 到達の見通し**: T2V のクリーン実測 2.53x が「演算律速のハードウェア素比」を
  表す。B が棄却された以上、**同一計算内容の制約（DLS-003）の下で CFG 条件の
  2.0 到達は現実的でない**。A/C は 11.33x → 2.6x のような大幅回復に意味があるが、
  2.0 という線そのものは CFG 条件では届かない可能性が高い

## 6. 留保

1. 本 PoC は GEMM 単体のマイクロベンチで、attention（AOTriton）や正規化・
   要素演算は対象外。ただし transformer 時間の支配項は GEMM であり、
   attention も系列長 2 倍で 2 倍以上（O(L²) 成分あり）になるため結論は保守側
2. TunableOp 無効で測ったため絶対値は調律後の本番経路と異なる。ratio 比較が目的で
   あり、調律は演算律速/重み読み律速の別を変えない
3. N=261 の小トークン帯では ratio 1.14〜1.27 の重み読み律速領域が実在する。
   極端に短い系列だけを扱う経路があればバッチ化に意味が出るが、
   実運用の T2I/T2V/I2V は N≥672 帯が支配的

## 7. 検証根拠（再現手段）

```
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host \
  -e PYTORCH_TUNABLEOP_ENABLED=0 \
  -v /home/kabayan/workspace/cosmos3:/workspace -w /workspace \
  cosmos3-rocm72-diffusers:local \
  bash -lc "python3 scripts/probe_cfg_batching_gemm.py --out result/cfg_batch_probe/gemm_bf16.json"
```
