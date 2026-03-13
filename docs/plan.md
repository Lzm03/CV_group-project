# MVP Implementation Plan

## Phase 1 — Core perception
- webcam input
- scene summary limited to the three supported demo objects: pen / paper / cell phone
- grasp-target support limited to pen / paper / cell phone for the current demo setup
- hand landmark extraction
- draw debug overlay

## Phase 2 — Natural voice interaction
- scene summary from natural spoken English
- spoken target selection
- target-relative direction answer using clock positions
- periodic spoken guidance every 10 seconds
- spoken grasp-status query handling

## Phase 3 — Demo hardening
- speech cooldown
- target lock / best-box selection
- low-light and missed-detection fallback handling
- record evaluation videos

## Evaluation ideas
- detection success rate
- scene-summary correctness
- spoken intent recognition correctness
- direction-answer correctness
- average guidance steps to reach target
- time-to-grasp proxy
- end-to-end latency
- failure cases

## Suggested 3-person split
- Person A: object detection + hand tracking
- Person B: voice interaction + audio + demo system integration
- Person C: evaluation + report + slides
