# DLS active エントリ

## DLS-003
- **date**: 2026-07-26
- **what**: 計算省略系の高速化（TeaCache 等の近似キャッシュ、INT8 量子化 attention 含む）を対論文同一条件比較の本線から除外する。TeaCache は速度成果としてではなく「出力が非適用時とどの程度異なるか」の品質差定量評価のみ実施する
- **why**:
  - origin: user_request
  - business: 本プロジェクトの対外的価値は「元記事と同じ計算内容・同一条件での実測比較」にあり、計算を省略した数値は比較として成立しない。一方で TeaCache の品質影響の実データは今後の判断材料として価値がある
- **where**: docs/cosmos3_rocm_further_speedup_reassessment_20260726.md §5-③⑤・§6、README.md（対外比較主張の範囲）
- **sources**: .claude/.dls/raw/20260726_chat_teacache_scope_decision.md
- **requested_by**: ユーザー
- **depends_on**: DLS-001, DLS-002
- **affects**: DLS-002（「残余候補を計算省略系+スタック更新に絞る」のうち、計算省略系を本線から外す方向に限定を上書き）
- **rejected_hypothesis**:
  - target: DLS-002（残余候補の絞り込みのうち計算省略系の本線採用可能性）
  - hypothesis: TeaCache 等の近似キャッシュは等価性判断①（クオリティ非劣化なら数%誤差許容）の範囲内なら本線の速度成果として採用できる
  - reason: ユーザー却下（2026-07-26）。等価性判断①は「同じ計算の数値誤差」の許容であり、「計算内容の省略」は元記事との比較前提を壊すため範囲外
- **rejected_alternatives**:
  - TeaCache 品質検証付き本線導入（レポート §5-③）: 上記理由で棄却
  - SageAttention INT8 の本線導入（レポート §5-⑤）: 同じく計算内容を変えるため本線不可。品質差評価の対象にも現時点では含めない（gfx1151 未検証で導入コスト大）
- **commits**:
  - baseline: b37cfb7
  - impl: 51c1b4e
- **assumption**: TeaCache 品質差評価は既存推論スクリプトへの非侵襲な追加（別ブランチ or フラグ分離）で実施でき、本線ベンチマーク環境を汚さない（confidence: high）

## DLS-002
- **date**: 2026-07-26
- **what**: 「vLLM（PagedAttention）統合で Policy Model サンプリングを高速化できる」仮説（v2.5 提案書の本命案）を棄却し、サンプリング高速化の残余候補を「計算の省略系（TeaCache 等）+ スタック更新の小幅改善」に絞る
- **why**:
  - origin: implementation
  - business: 残余高速化余地の再調査（DLS-001）の過程で、v2.5 提案書が本命としていた vLLM 統合の前提が成立しないことが判明したため、無効な方向への工数投入を防ぐ
  - constraint: PagedAttention の利益源（KV キャッシュ断片化解消 / autoregressive decode / prefix 共有）は、バッチ 1・固定長 74k トークン・毎ステップ全再計算・decode ゼロの DiT サンプリングには存在しない。vLLM 公式の DiT 対応（vllm-omni）自体が PagedAttention を使っていない
- **where**: docs/cosmos3_rocm_further_speedup_reassessment_20260726.md §3、docs/cosmos3_rocm_policy_optimization_proposals.md（v2.5 提案の本命案が対象）、temp_src/packages/vllm-cosmos3/
- **sources**: .claude/.dls/raw/20260726_discussion_further_speedup_scope.md、docs/cosmos3_rocm_further_speedup_reassessment_20260726.md
- **requested_by**: 自己判断（調査結果に基づく）
- **depends_on**: DLS-001
- **affects**: なし
- **rejected_hypothesis**:
  - target: （DLS 化前の判断: v2.5 提案書 docs/cosmos3_rocm_policy_optimization_proposals.md §2-①）
  - hypothesis: vllm-cosmos3 統合で Transformer 順伝播を PagedAttention 最適化パスに移植すれば 1.0s/it 以下に短縮できる
  - reason: (1) ローカルの vllm-cosmos3 は Reasoner（VLM）用公式プラグインで Policy の DiT に非適用、(2) PagedAttention の利益源が本ワークロード構造に存在しない、(3) vLLM 公式 DiT 対応（vllm-omni）も PagedAttention 不使用で高速化源は演算子融合（本環境では AOTriton FA として適用済み相当）。2026-07-26 Web 調査 + ローカル確認による
- **rejected_alternatives**:
  - vllm-omni の演算子融合転用: RDNA/gfx1151 での diffusion engine 動作報告が現時点で皆無のため dormant（RDNA 実証が出たら再評価）
- **commits**:
  - baseline: b37cfb7
  - impl: 51c1b4e
- **assumption**: TeaCache 系の近似キャッシュが 2026-06-12 等価性判断①（クオリティ非劣化なら数%誤差許容）の範囲に収まる（confidence: medium。導入時に同一 seed 品質比較で検証する）

## DLS-001
- **date**: 2026-07-26
- **what**: gfx1151 での Cosmos3 Policy 推論について、生成条件（30 steps / 640x480 / 17f）と同期総和測定を維持したまま、残余高速化余地の再調査結果を 1 つのドキュメントとして docs/ に残す
- **why**:
  - origin: user_confirmed
  - business: 最終報告（総実行時間 41.66 秒、2026-06-14）から約 6 週間経過し、ROCm / AOTriton / PyTorch 等の更新により「メモリ帯域の物理限界（1.127s/step）」結論の前提が変わった可能性を確認したい。また未完の可能性がある vLLM-cosmos3 統合の残余価値も未整理
  - constraint: 「同一条件・同期総和での対論文比較」がプロジェクトの当初価値であり、これを壊す調査スコープは採らない
- **where**: docs/（新規調査ドキュメント）、scripts/run_cosmos_framework_policy_rocm.py（調査対象の推論経路）、docs/cosmos3_rocm_performance_limit_reassessment.md（再評価対象の結論）
- **sources**: .claude/.dls/raw/20260726_discussion_further_speedup_scope.md
- **requested_by**: ユーザー
- **depends_on**: なし（初エントリ）
- **affects**: なし
- **rejected_alternatives**:
  - 案B（制約緩和込み全方位ロードマップ）: final report §3 に既出で新規性が薄く、同一条件比較というプロジェクト当初価値と矛盾するため dormant
  - 案A+B（統合ドキュメント）: 主軸がぼけ、B 部分は既出情報の再掲になるため dormant
  - 案C（何もしない / YAGNI）: 約 6 週間分の環境更新が「物理限界」結論に未反映のまま凍結されるため dormant
- **commits**:
  - baseline: b37cfb7
  - impl: 51c1b4e
- **assumption**: ROCm / AOTriton の 2026-06-14 以降の更新に gfx1151 向け attention / GEMM の性能改善が含まれる（confidence: medium。調査自体がこの前提の検証を兼ねる）
