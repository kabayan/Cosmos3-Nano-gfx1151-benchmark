# 必須ツール（DLS-124 主軸: token-efficient な CLI を優先、MCP は副）

init.py 実行時に PATH 検査 → 不在ならインストール提案（`auto_install_cmd` がある必須ツールはデフォルト Y で `npm i -g` / `uv tool install` を実行）。

## cocoindex

AST ベースのコードベース意味検索（70% トークン削減、SOURCE-029）。CLI 主軸 (`ccc`)、MCP は副。

### まず uv をインストール（前提）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### CLI（推奨、DLS-124）

```bash
uv tool install --upgrade cocoindex-code --prerelease explicit
ccc init    # プロジェクト初期化
ccc index   # インデックス構築
ccc search "認証ロジック"   # 自然言語検索
```

`pipx install cocoindex-code` でも可。

### MCP（オプション）

```bash
claude mcp add cocoindex-code \
  -- uvx --prerelease=explicit --with \
  "cocoindex>=1.0.0a16" \
  cocoindex-code@latest
```

## serena 設定（オプション）

```bash
claude mcp add serena -- uvx --from git+https://github.com/oraios/serena serena start-mcp-server --context ide-assistant --project $(pwd)
/init
```

## playwright-cli

URL 本文取得 + ブラウザ自動化（公式 @playwright/cli、token-efficient、DLS-124 主軸）。

### CLI（推奨）

```bash
npm install -g @playwright/cli@latest
playwright-cli install --skills    # Claude Code skills 連携（init.py auto_install で実行済み、DLS-135）
playwright-cli open <URL>          # ブラウザで URL を開く
playwright-cli snapshot            # ページの a11y スナップショット取得
```

公式 README 曰く CLI は MCP より低トークン（large tool schemas を context に載せない）。

## codex

codex CLI による異モデルコードレビュー（`/dls-codex-review` skill で利用）。Opus 4.7 の手抜き検知 + コミット前自動レビューが動機（DLS-154）。

### CLI

codex は npm / uv では配布されていない。GitHub release から該当 OS バイナリを取得して PATH に配置する。

```bash
# https://github.com/openai/codex/releases から該当 OS バイナリをダウンロード
# 例: Linux x86_64
# tar.gz を展開し、codex バイナリを ~/.local/bin/ などに配置 (PATH の通った場所)

codex --version       # codex-cli 0.128.0 以上で `codex review` サブコマンド提供 (DLS-132 PoC #4 動作確認実績あり)
codex review --help   # オプション確認

# 動作確認
codex review --uncommitted
```

`/dls-codex-review` skill が呼ばれた時のみ必要（optional）。init.py は不在検出時に警告のみ表示し、続行はブロックしない。codex 不要なプロジェクトでは無視してよい。

## playwright MCP（オプション）

```bash
claude mcp add playwright npx @playwright/mcp@latest
```

DLS-124: CLI を主軸とし MCP は副（CLI 不在時のみ）。

## chrome dev tool（重量・オプション、DLS-121→124 で副降格）

```bash
sudo apt update
sudo apt install -y wget
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -f ./google-chrome-stable_current_amd64.deb
rm ./google-chrome-stable_current_amd64.deb
claude mcp add chrome-devtools npx chrome-devtools-mcp@latest -- --headless true
```

プロファイル衝突あり、起動が重い。playwright-cli が使えない場合のみ。

# chrome dev tool(Dokcerコンテナ版) 
git clone https://github.com/fcf-koga/chrome-mcp-devcontainer.git
cd chrome-mcp-devcontainer

vi .debcontainer/Dockerfile
# ビルド引数としてプロキシ情報を受け取る
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# それらの引数をコンテナ内の環境変数として設定する
ENV http_proxy=$HTTP_PROXY
ENV https_proxy=$HTTPS_PROXY
ENV no_proxy=$NO_PROXY
:


docker build -t chrome-mcp-headless \
  -f .devcontainer/Dockerfile \
  --build-arg HTTP_PROXY="http://192.168.0.181:3128/" \
  --build-arg HTTPS_PROXY="http://192.168.0.181:3128/" \
  --build-arg NO_PROXY="localhost,127.0.0.1" \
  .

docker run -d --name chrome-mcp --network host chrome-mcp-headless sleep infinity
docker exec -d chrome-mcp google-chrome \
  --headless \
  --no-sandbox \
  --disable-gpu \
  --disable-dev-shm-usage \
  --remote-debugging-port=9222 \
  --remote-debugging-address=0.0.0.0 \
  --user-data-dir=/tmp/chrome-debug

claude mcp remove chrome-devtools-browser
claude mcp add chrome-devtools-browser -s user -- npx -y chrome-devtools-mcp@latest --browser-url=http://localhost:9222

# ワークフロー
## spec-workflow(新規) 
### claude code UI
https://github.com/siteboon/claudecodeui

#### インストール

##### nodeが古い場合
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm use 20
nvm alias default 20
nvm install 20
rm -rf node_modules package-lock.json
npm install

##### インストール
git clone https://github.com/siteboon/claudecodeui.git
cd claudecodeui
npm install
cp .env.example .env
npm run dev

#### WebUI
http://localhost:3001

### セットアップ
claude mcp add spec-workflow npx @pimzino/spec-workflow-mcp@latest $(pwd)
#### WebUI
npx -y @pimzino/spec-workflow-mcp@latest $(pwd) --dashboard
### 使い方
1. @docs/sped.md, @docs/rules.md, @docs/envs.mdを基に spec を作成して
    - ここでのspec名を覚えておく
2. webuiでタスク確認
3. spec に 機能：ｘｘを追加して更新
ｘｘの Steering ドキュメントを更新
4. webuiでタスク確認。足りなければ３へ
5. task.mdを参考に task 〇〇を実装。モックテスト不可、Dockerコンテナ内実施必須。100%テストpassは必要。 @docs/rules.md を厳守していること。簡易的なテストは不可。テストデータの変更による100%成功は不可

## cc-sdd（既存拡張）

### セットアップ

#### claude-code-viewer
PORT=3400 npx @kimuson/claude-code-viewer@latest

#### インストール
npx cc-sdd@latest --lang ja

### 使い方
- https://github.com/gotalab/cc-sdd/blob/main/tools/cc-sdd/README_ja.md
- https://zenn.dev/kokushing/articles/7468d5f195e54c

1. Steering Documents の作成 (/kiro:steering)
2. Specs テンプレートの作成 
- /kiro:spec-init 新規追加機能
- /kiro:steering 
3. requirements.md の作成 (/kiro:spec-requirements webui-progress-display)
    - .kiro/xx/requirement.md を確認
4. /kiro:steering webui-progress-display
5. design.md の作成 
/kiro:spec-design webui-progress-display

- /kiro:spec-design webui-progress-display -y
6. /kiro:steering 
7. tasks.md の作成 (/kiro:spec-tasks)
8. タスク番号を指定して実行 (/kiro:spec-impl)
- /kiro:spec-impl <specname> タスク1から順番に進めて
9. /kiro:spec-status webui-progress-display # 進捗を確認


# 初期
コミットして https://github.com/kabayan/dls にpush sshキーはid_ed25519 emailはkabayan@adlibjapan.jp userはkabayan

# フロー例
## specbase.mdを作成
概要仕様を作成

## specを作成
@docs/specbase.md を基に specを作成。各種ドキュメントは日本語で作成。

## rules.md を反映
@docs/rules.md を specに反映

## steering documentsを作成
steering documents を日本語で作成

## spec を更新後にやること
タスクも spec の更新に対応させて

# github tip
https://github.com/kabayan/dls をカレントにpull。サブフォルダは作らない。sshキーはid_ed25519