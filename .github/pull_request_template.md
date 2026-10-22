## 変更点

-

## 確認

- [ ] `python scripts/check_chapters.py` / `check_code_quotes.py` / `check_python_examples.py` / `check_versions.py` / `check_docs_match.py` が通る
- [ ] `python -m pytest` と、`examples/` の `uv run pytest`・ruff が通る
- [ ] `mkdocs build --strict` が通る
- [ ] 挙動や仕様の説明に、公式ドキュメントへのリンクか、`examples/` のコードとテスト（src コメント付きの引用）を根拠として添えた（CI では確かめられないので、レビューで見る）
- [ ] 従量課金の API を使う手順は「任意」と明記し、ローカルのモデルで進める道と、費用の上限・止め方を書いた（該当する場合）
- [ ] 実在の会社名・個人名・社内 URL・ローカルのユーザーパス・API キーを書いていない。ほかの教材やリポジトリを参照していない
