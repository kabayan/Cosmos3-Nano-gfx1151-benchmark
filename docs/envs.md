# 環境ルール

## 禁止事項

- [ ] ローカル環境、実行中コンテナへのpipインストール禁止（Dockerfile修正などで対応）

## 必須実施事項

### 実行環境
- [ ] LLMはAPI経由で利用（直接利用禁止）
- [ ] LLMサーバーは既存のサーバーを利用
- [ ] LLMサーバーへのアクセスはopenai api互換エンドポイントを使う

## LLM設定

### 利用可能LLM
- モデル名は異なる場合がある
   ```bash
   curl -s http://192.168.1.18:8000/v1/chat/completions \
     -H "Authorization: Bearer EMPTY" \
     -H "Content-Type: application/json" \
     -d '{ 
       "model": "qwen3-14b-awq", 
       "messages": [{"role": "user", "content": "日本の首都は？"}],
       "max_tokens": 64, 
       "temperature": 0.2,
       "chat_template_kwargs": {"enable_thinking": false}
     }'
   ```
