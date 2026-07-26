# DLSエントリ ひな形

このファイルをコピーして使う。必須フィールドは必ず埋める。

---

## 【コピー用】フルテンプレート

```
## DLS-{連番}
- **date**: YYYY-MM-DD
- **what**: （Why/Whatのみ。Howは含めない）
- **why**:
  - origin: （判断の出所。以下のいずれか）
    - `user_request` — ユーザーが明示的に要求した
    - `user_confirmed` — 実装者が提案しユーザーが承認した
    - `implementation` — 実装中に技術的に判明・決定した（ユーザー未確認）
  - business: （ビジネス上の動機・ユーザーの課題）
  - constraint: （技術的制約・外部要因。止むを得ずHowに触れる場合のみ）
- **where**: （影響するファイル、モジュール、機能範囲）
- **sources**: （原本ファイルのパス。例: `.claude/.dls/raw/20260323_hearing_xxx.md`）
- **requested_by**: （クライアント名 / ユーザー名 / 自己判断）
- **depends_on**: （DLS-XXX）
- **affects**: （DLS-XXX, DLS-XXX）
- **supersedes**: （DLS-XXX）
- **rejected_hypothesis**: （supersedesがある場合は強く推奨）
  - target: （棄却対象のDLS-XXX）
  - hypothesis: （棄却された仮説の1行要約）
  - reason: （棄却の根拠。実証的: テスト失敗 / ユーザー拒否 / 実装試行など）
- **rejected_alternatives**:
  - （案A）: （dormant化した理由。今回採らなかった理由）
  - （案B）: （dormant化した理由）
- **commits**: （新規エントリ必須、既存エントリ免除）
  - baseline: （判断時のコード状態のSHA。任意。バックトラックの戻り先起点）
  - impl: （判断を反映した実装コミットSHA。複数可、カンマ区切り。戻す範囲）
  - reject_evidence: （棄却された案を実装試行した痕跡のSHA。任意。dormant 再評価時の参照）
- **assumption**: （未検証の前提。confidence: high/medium/low）
```

---

## 【コピー用】最小テンプレート（必須フィールドのみ）

```
## DLS-{連番}
- **date**: YYYY-MM-DD
- **what**:
- **why**:
  - origin: （user_request / user_confirmed / implementation）
  - business:
- **where**:
- **sources**:
```

---

## フィールド記入ガイド

### `what` — 何を決めたか（最重要）

外から見たときに何を満たすべきかを書く。**How（実装手段）は含めない。**

| ✅ 良い例 | ❌ 悪い例（How混入） |
|---|---|
| セッションの有効期限を30分に制限する | Redisを使いTTLを1800秒に設定する |
| 検索結果を関連度順で返す | ElasticsearchのBM25スコアでソートする |
| 画像アップロードの最大サイズを5MBとする | multerのlimitsをbytes: 5242880に設定する |

### `why` — なぜそう判断したか

3つのサブフィールドで構成:

**`origin`（必須）** — 判断の出所を明示する:

| origin | 意味 | 例 |
|---|---|---|
| `user_request` | ユーザーが明示的に要求した | 「PerHead方式で実装して」 |
| `user_confirmed` | 実装者が提案しユーザーが承認した | 「PerHead方式を提案→承認」 |
| `implementation` | 実装中に技術的に判明・決定した（ユーザー未確認） | 「ROCm pagefaultで回避不可と判断」 |

**`business`（必須）** — ビジネス動機・ユーザー課題

**`constraint`（任意）** — 技術的制約・外部要因。`business` のみで説明できる場合は省略可。

### `sources` — 原本へのパス（最重要）

DLSエントリは「LLMの現時点の解釈」に過ぎない。原本が消えたら再解釈できない。
必ず `.claude/.dls/raw/` 以下の実ファイルパスを記載する。

### `depends_on` / `affects` — 判断の連鎖

- `depends_on`: このエントリの判断が前提としている他のエントリ
- `affects`: このエントリが変更されると見直しが必要になるエントリ
- 判断追加時に既存エントリの `affects` も更新する

### `supersedes` — 過去の判断を覆す場合

過去のエントリを上書き・削除しない。新エントリに `supersedes: DLS-XXX` を記載し、
古いエントリを `archive.md` に移動する。

### `commits` — 関連 git コミット（バックトラックの戻り先）

DLS は判断のログだが、判断時のコード状態 / 反映差分 / 棄却試行の痕跡を構造化しないと
バックトラック（NG 仮説から手前に戻る）時に「どの commit に戻ればよいか」を人が grep
で探す必要がある。サブフィールドで構造化する:

| サブフィールド | 用途 | 必須性 |
|---|---|---|
| `baseline` | 判断時点のコード状態 SHA。「ここに戻れば判断当時のコンテキストが復元できる」起点 | 任意 |
| `impl` | 判断を反映した実装コミット SHA（複数可）。「ここまで戻せば判断を巻き戻せる」範囲 | 任意 |
| `reject_evidence` | 棄却された案を実装試行した痕跡 SHA。dormant 再評価時に「どこまでやって NG だったか」を見るための参照 | 任意 |

**適用範囲（DLS-144 軟上限と同様の段階導入）**:
- 新規エントリ: 全サブフィールドが任意だが、判断とコードが直結する場合は少なくとも 1 つを記入する
- 既存エントリ（dls_core_version 1.14.0 未満で起票されたもの）: 免除。protected_paths 配下のため migration では補完不可。遡及補完が必要ならユーザーが手動で active.md を編集する

記入タイミング:
1. 判断確定時に `baseline: <現在 HEAD>` を記録（任意）
2. 実装コミット後に `impl: <SHA>` を追記
3. dormant 案を実装試行して NG なら `reject_evidence: <試行 SHA>` を追記し、`rejected_hypothesis.reason` で参照

### `assumption` — 未検証の前提

confidence レベルで管理する：
- `confidence: high` — ほぼ確実だが未確認
- `confidence: medium` — 確認が必要
- `confidence: low` — 要早期検証。放置するとリスクになる
