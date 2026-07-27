# conditioning の 189 フレーム全エンコード疑いの検証（原本）

- 日時: 2026-07-26
- 検証者: CC（静的コード読解 + 既存計測アーティファクトの突合）
- 対象仮説: `.claude/.dls/raw/20260726_chat_baseline_audit_and_scope_estimation.md` 未検証仮説 2
  「conditioning 81.57 秒 ÷ 0.43 秒/フレーム ≈ 189 フレーム分。入力観測動画の全フレームを
  VAE エンコードしているのではないか」
- 結論: **仮説は成立する（確定）**。ただし「入力観測動画の全フレーム」ではなく
  「生成長 `num_frames=189` にパディングされたテンソル全体」を VAE エンコードしている。

## 1. 数値の裏取り

`result/mainline_full_v4_20260726/action_policy_robot/sample_args.json`（実行時の解決済み引数）:

```
"condition_frame_indexes_vision": [0, 1]
"condition_video_keep": "first"
"num_frames": 189          ← policy モードの既定値
"fps": 5, "resolution": "720", "num_steps": 30
```

`num_frames: 189` は `temp_src/cosmos_framework/inference/defaults/policy/sample_args.json` の
既定値がそのまま解決されたもの（入力 JSON `inputs/omni/action_policy_robot.json` は num_frames を
指定していない）。推定値 189 と実測の一致は偶然ではない。

`result/mainline_full_v4_20260726/policy_stage_sync_profile.json` の集計（measured フェーズ）:

| stage | n | 合計秒 |
|---|---|---|
| generate_batch_sync | 1 | 124.906 |
| generate_samples_from_batch_sync | 1 | 117.253 |
| prepare_inference_data_sync | 1 | 81.615 |
| **get_data_and_condition_sync** | 1 | **81.575** |
| get_velocity_sync | 30 | 35.539 |
| decode_sync | 1 | 7.348 |

`prepare_inference_data` 81.615 秒のうち 81.575 秒（99.95%）が `get_data_and_condition`。
これは総時間 124.906 秒の **65.3%** を占める。

## 2. コード経路（全フレームがエンコードされる仕組み）

1. `cosmos_framework/inference/inference.py` L553-561
   `num_condition_latent_frames = max(condition_frame_indexes_vision) + 1 = 2`
   `max_frames = tokenizer.get_pixel_num_frames(2)`。Wan2.2 VAE は
   `(n-1)*4+1`（`model/vfm/tokenizers/wan2pt2_vae_4x16x16.py` L1653-1654）なので **5 ピクセルフレーム**。
   → `load_conditioning_video` はここで入力動画を先頭 5 フレームに切り詰めている。
   **入力動画の全フレームを読んでいるわけではない**（仮説の文言はここだけ不正確）。

2. `cosmos_framework/inference/vision.py` L106-127 `build_conditioned_video_batch`
   ```python
   video_data = torch.zeros(1, 3, num_frames, h, w)       # num_frames = 189
   t_fill = min(t_cond, num_frames)                        # = 5
   video_data[0, :, :t_fill] = conditioning_frames[...]    # 実データは 5 フレームのみ
   video_data[0, :, t_fill:] = video_data[0, :, t_fill-1:t_fill].expand(...)  # 残り 184 は最終フレームの複製
   ```
   → **ここで 5 フレームが 189 フレームに水増しされる**。

3. `cosmos_framework/model/vfm/omni_mot_model.py` L2864-2866 `get_data_and_condition`
   ```python
   x0_tokens_vision = [self.encode(v).contiguous().float() for v in raw_state_vision]
   ```
   → 189 フレームのテンソルをまるごと VAE エンコード。conditioning インデックスによる
   スライスは一切入らない。48 latent frames （(189-1)/4+1）が生成される。

4. `omni_mot_model.py` L1667-1678（推論時のノイズ初期化）
   ```python
   noise_i = cond_mask * x0_token + (1.0 - cond_mask) * pure_noise_i
   ```
   `cond_mask` は「1 = clean/conditioning, 0 = noisy/generation」（L892 のドキュメント文字列）。
   条件付けは latent frame [0, 1] のみなので、**latent frame 2〜47 の x0 値は 0 倍されて捨てられる**。
   使われるのは shape だけ。

## 3. 無駄の量

- 実際に必要なエンコード: ピクセルフレーム 0〜4（latent frame 0〜1）= 5/189 フレーム
- 実際に走っているエンコード: 189 フレーム
- **約 97.4%（184/189 フレーム）が計算後に 0 倍されて破棄されている**

Wan2.2 VAE は `CausalConv3d` + `feat_cache` によるストリーミング因果構造
（`wan2pt2_vae_4x16x16.py` L43-74, L149-223）なので、latent frame k は先行ピクセルフレームのみに
依存する。したがって**先頭 5 フレームだけをエンコードしても latent frame [0,1] はビット一致するはず**
（この点は未実証。実装時に同一 seed で latent 一致を確認する必要がある）。

## 4. 見積もられる効果

81.575 秒がすべて encode だと仮定した線形外挿で 81.575 × 5/189 ≈ 2.2 秒。
削減幅は約 79 秒。conditioning 込み総時間は 124.91 秒 → 約 45.5 秒（2.7 倍速）になる。

注意（過大評価しないための留保）:
- 81.575 秒には `_normalize_video_databatch_inplace` / `_augment_image_dim_inplace` /
  `_remove_padding_from_latent` / action 正規化も含まれる（ただし別途計測されている
  `normalize_video_databatch_sync` は 0.010 秒なので encode 支配は確実）
- VAE エンコードはフレーム数に完全線形とは限らない（固定コスト・カーネル起動・キャッシュ効果）
- 上記はいずれも**実測していない外挿**である

## 5. headline 数値（41.66 秒）への影響

**影響しない**。`--policy-condition-cache` は measured フェーズの conditioning を warmup 結果で
置換するため（DLS-005）、41.66 秒にはそもそも conditioning が含まれていない。
本件が変えるのは「conditioning 込みの正直な総和 124.91 秒」の側であり、
DLS-006 で参考値として併記した数字である。

## 6. DLS-003 との関係（計算省略系ではない）

DLS-003 は「計算内容の省略（TeaCache 等の近似キャッシュ）を本線から除外する」と決めた。
本件は**近似ではない**: 破棄される latent は `cond_mask=0` により厳密に 0 倍されるため、
エンコードを省いても生成結果はビット一致する（§2-4 の因果性前提が成り立てば）。
したがって DLS-003 の制約対象外と考えられるが、フレームワーク本体（temp_src /
/tmp/cosmos-framework）への改変を伴うため、実施可否はユーザー判断を要する。

## 7. 未実施 / 次の検証

- [ ] 先頭 5 フレームのみエンコードした latent frame [0,1] が 189 フレームエンコード時と
      一致することの実測（因果性の実証）
- [ ] encode 単体の実測時間（189 フレーム vs 5 フレーム）。線形外挿の検証
- [ ] 実装方針の選択（未着手・ユーザー判断待ち）

---

## 【追記 2026-07-27】本ノートの §2-§4 は棄却された

実測プローブにより、encode 入力は 189 フレームではなく **17 フレーム**
（`[1,3,17,544,736]`）であることが確定した。policy は `inference.py` L513-517 で
action 系モードとして早期 return し `build_action_batch` を通るため、
本ノートが追った `build_conditioned_video_batch`（`num_frames=189` でパディングする関数）には
**到達しない**。分岐の取り違えである。

- 棄却の経緯: `.claude/.dls/raw/20260727_doc_conditioning_probe_refutes_189frame_claim.md`
- 訂正後の実測: `.claude/.dls/raw/20260727_doc_conditioning_probe_v2_measured.md`
- DLS エントリ: DLS-008（`rejected_hypothesis` に本仮説を記載）

本ノートは「その時点でそう考えた」記録として残置する（raw/ は追記のみ）。
§5（headline 41.66 秒に影響しない）と §6（近似ではないため DLS-003 の対象外）の
結論自体は訂正後も維持されている。
