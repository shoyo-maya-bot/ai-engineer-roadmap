# Copilot への指示（ai-engineer-roadmap）

このリポジトリで教材や演習のコードを書く・直すときは、リポジトリ直下の [AGENTS.md](../AGENTS.md) に従ってください。要点:

- このリポジトリだけで完結させる（ほかの教材・リポジトリ・社内のツールを参照しない）
- 章は CONTRIBUTING.md の型（到達目標・何か・なぜ必要か・どうやるか・演習・もし〇〇がなかったら？・理解度チェック）で書き、解答は `answers/` に分ける
- 実行できるコードは `examples/` に実物とテストを置き、本文には `<!-- src: examples/... -->` を付けて引用する
- テストと CI から LLM や外部の API を呼ばない（`FakeLLM` を使う）。従量課金の API を使う手順は「任意」と明記する
- サンプルは架空のものだけ。実在の会社名・個人名・メールアドレス・ローカルのパス・API キーを書かない
- 変更後は `scripts/check_*.py`・`python -m pytest`・`mkdocs build --strict`・`examples/` の `uv run pytest` と ruff を実行する
