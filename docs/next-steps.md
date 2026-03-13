# Next Steps

## 1. Install dependencies
Recommended:
```bash
cd group-project-mvp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run the app
```bash
cd group-project-mvp/src
python main.py
```

## 3. Current demo behavior
- The system is query-driven rather than continuously real-time
- Press `v` to ask one natural English question
- The app captures one frame, analyzes it, and answers once
- Speech output can use MiniMax TTS with a custom `voice_id`
- If MiniMax fails or is not configured, it falls back to macOS `say`

## 3.1 MiniMax setup
You can either export these before running:
```bash
export MINIMAX_API_KEY=your_api_key
export MINIMAX_GROUP_ID=your_group_id
export MINIMAX_VOICE_ID=your_voice_id
```
or put them in `group-project-mvp/.env`.
The app now loads `.env` automatically on startup.
You can also set `minimax_voice_id` in `src/config.py`.

## 4. Example queries
- What objects are in front of me?
- I want to pick up the cup.
- Where is the cup?
- Did I get the cup?

## 5. What to improve next
- better grasp-status estimation
- evaluation and logging
- report figures and screenshots
- optional speech activity indicator on screen
