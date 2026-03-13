# Demo Checklist

## Before demo
- Put 2-3 supported demo objects on a desk: pen, paper, phone
- Use stable lighting
- Keep clutter low
- Verify webcam, microphone, Whisper CLI, and MiniMax TTS config all work

## Query-snapshot demo flow
1. Start app
2. Live camera is only for preview
3. Press `v`
4. Say: `What objects are in front of me?`
5. The system captures ONE frame and answers using only that snapshot
6. Press `v` again
7. Say: `I want to pick up the cup.`
8. The system captures ONE new frame and answers where the cup is
9. Move your hand closer
10. Press `v` again
11. Say: `Did I get the cup?`
12. The system captures ONE new frame and answers yes/no

## Design choice
This MVP is not real-time tracking. It is query-driven.
Every user question triggers a fresh snapshot, then object detection, hand analysis, and one spoken response.
This reduces jitter and improves reliability for demo use.

Note: `person` is filtered out from scene summaries to reduce noise.
Small thin objects like pens are not reliably detected by the default general-purpose YOLO model.
Paper is also less stable than common rigid household objects. Phone is expected to be the most reliable among the three supported targets.
