# Microgame Rush: a WarioWare-style cue system

A WarioWare-style game isn't scored with one long loop. It uses a **run** of short cues that land exactly on bar lines. `src/tracks/microgame.py` renders those cues with Canopy Orchestra's synthesized instruments, and `player/microgame.html` plays a demo run. Pick **Microgame Rush** in the Soundtrack menu.

## The run

```
interlude (2 bars) ─▶ microgame (2 bars) ─▶ win | lose (1 bar) ─┐
     ▲                                                          │
     ├── every 4th game: speedup (2 bars), next tempo tier ◀────┤
     ├── game 12: boss_intro (2) ─▶ boss (4-bar loop ×2) ─▶ boss_win (2) ─▶ clear (2) ─▶ next tier
     └── lives = 0: gameover (2 bars)
```

- **Tiers:** 120, 144, 160 and 180 BPM. Every cue is rendered at every tier, so speed-ups keep their timbre. There's no time-stretching.
- **Harmony:** everything is in C major or A minor. Each loop starts on C or Am, and the interlude ends on G7, so any cue can follow any other.
- **The 8 microgame styles:** chiptune, funk, mariachi, surf, jungle marimba (the Canopy link), space disco, toy box and polka. Each has its own 2-bar hook, and the hook is the loudest part of its cue.
- **Loops** (interlude, microgames, boss) are circular, so they loop seamlessly, and each has a `_tail` ring-out for when it stops. **One-shots** carry their own tail; start the next cue at the one-shot's `seconds` and let the tail ring over the join.
- **Loudness:** loops sit at −18 LUFS and jingles at −16 LUFS.

## Files

`python3 -m tracks.microgame` writes `out/microgame/tier0…tier3/<cue>.ogg` (and `<loop>_tail.ogg`), plus `cues.json`:

```json
{"tiers":[120,144,160,180], "grammar":{"speedup_every":4,"boss_at":12,"lives":4},
 "cues":{"tier0":{"bpm":120,"cues":{"mg_funk":{"kind":"loop","bars":2,"samples":192000,"seconds":4.0}, "...":{}}}}}
```

`--web` also writes the player's sprites to `player/microgame/`: one AAC file per tier, with a sync impulse at sample 1000 so codec padding can't shift the cues, base64 copies, and `microgame.json` with the offsets.

## Hooking up a game (Godot, ¡Rápido!, anything)

- Drive it from the audio clock. Start each cue exactly at the previous cue's end time (`bars × 240 / bpm`).
- **Resolve once.** A microgame's result can arrive at any time during its 2 bars. Decide it about 90 ms before the bar line, then start `win` or `lose` on the line. That matches ¡Rápido!'s `microgame.gd` rule that `resolve()` fires exactly once.
- Play an instant success or fail sound when the player resolves early (the demo uses `events.js` `collect` and `hurt`). The jingle still waits for the bar line.
- A microgame that needs longer can loop its cue for extra passes. Boss stages do this.

The browser API mirrors that flow: `CanopyGame.start()`, `press()`, `resolve(win)`, `getState()`, `setLanguage('es')`, `audition(cue, tier)`, and `on('cue'|'beat'|'microgame'|'result'|'speedup'|'boss'|'gameover', fn)`.
