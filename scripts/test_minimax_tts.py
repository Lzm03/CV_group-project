#!/usr/bin/env python3
from pathlib import Path
import json
import os
import subprocess
import tempfile

import requests
from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')

    api_key = os.getenv('MINIMAX_API_KEY', '')
    group_id = os.getenv('MINIMAX_GROUP_ID', '')
    voice_id = os.getenv('MINIMAX_VOICE_ID', '')
    model = os.getenv('MINIMAX_MODEL', 'speech-02-hd')
    text = os.getenv('MINIMAX_TEST_TEXT', 'Hello. This is a MiniMax voice test for the group project demo.')

    print('MINIMAX_API_KEY set:', bool(api_key))
    print('MINIMAX_GROUP_ID:', group_id)
    print('MINIMAX_VOICE_ID:', voice_id)
    print('MODEL:', model)
    print('TEXT:', text)

    if not api_key or not group_id or not voice_id:
        raise SystemExit('Missing MINIMAX_API_KEY, MINIMAX_GROUP_ID, or MINIMAX_VOICE_ID in .env')

    url = f'https://api.minimax.chat/v1/t2a_v2?GroupId={group_id}'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': model,
        'text': text,
        'stream': False,
        'voice_setting': {
            'voice_id': voice_id,
            'speed': 1.0,
            'vol': 1.0,
            'pitch': 0,
        },
        'audio_setting': {
            'sample_rate': 32000,
            'bitrate': 128000,
            'format': 'mp3',
            'channel': 1,
        },
    }

    print('\nPosting request to MiniMax...')
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print('HTTP status:', r.status_code)
    try:
        data = r.json()
    except Exception:
        print(r.text)
        raise

    print('Top-level keys:', list(data.keys()))
    print('JSON preview:', json.dumps(data, ensure_ascii=False)[:1200])

    audio_hex = data.get('data', {}).get('audio') or data.get('audio')
    audio_url = data.get('data', {}).get('audio_url') or data.get('audio_url')

    if audio_hex:
        audio_bytes = bytes.fromhex(audio_hex)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            f.write(audio_bytes)
            path = f.name
        print('Saved audio from hex to:', path)
        subprocess.run(['afplay', path], check=False)
        return

    if audio_url:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            path = f.name
        rr = requests.get(audio_url, timeout=30)
        rr.raise_for_status()
        Path(path).write_bytes(rr.content)
        print('Downloaded audio from URL to:', path)
        subprocess.run(['afplay', path], check=False)
        return

    print('No playable audio found in response.')


if __name__ == '__main__':
    main()
