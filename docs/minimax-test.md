# MiniMax TTS Test

Run this in the project venv:

```bash
cd ~/.openclaw/workspace/group-project-mvp
source .venv/bin/activate
python scripts/test_minimax_tts.py
```

What it checks:
- whether `.env` is loaded
- whether API key / group id / voice id are present
- whether MiniMax returns audio
- whether the returned audio can be played with `afplay`

Optional:
```bash
export MINIMAX_TEST_TEXT='Hello Jayden, this is a custom MiniMax voice test.'
python scripts/test_minimax_tts.py
```
