# Prompt: integrate the Canopy adaptive soundscape into a 2D jungle platformer

Copy everything below the line into the other AI, working in the game's codebase with the Canopy repository available.

---

You are integrating **Canopy**, an adaptive jungle soundscape, into a 2D side-scrolling platformer set in a jungle. The audio already exists; your job is the runtime system that plays it and reacts to gameplay. Do not re-compose or re-mix the audio files.

## 1. First, orient yourself

1. Identify the game's engine and its existing audio code (Unity, Godot, a web engine such as Phaser, GameMaker, a custom engine, or middleware like FMOD or Wwise). Everything below is engine-neutral. Use the engine's own tools where they fit.
2. Read `docs/ENGINE.md` in the Canopy repo. It is the spec: tempo, grid, file list and transition rules.
3. Read `player/index.html` in the Canopy repo. Its script is a complete, working reference implementation of the conductor in Web Audio. Port its logic rather than inventing new rules. The functions that matter are `chooseSection`, `dirStep`, `arrange`, `musicGain`, `applyStem`, `sectionChange`, `planBoundary` and `tick`, plus the `ORDER` and `WATER_BASE` tables.
4. Before writing code, write a short plan: which engine features you will use for sample-accurate scheduling, how you will load audio, and how level data will drive the parameters. Then implement.

## 2. What the audio is

- **Format:** 48 kHz stereo Ogg Vorbis. Tempo is 90 BPM in 4/4.
  - 1 beat = 32 000 samples; 1 bar = 128 000 samples (2.667 s).
  - Every loop is exactly 16 bars = 2 048 000 samples (42.667 s).
  - Loops are seamless and phase-locked: all layers of an area start on the same sample and must stay aligned forever.
- **Five areas.** Every bar 1 is a D chord and every bar 8 and bar 16 rests on an A chord, so any area can follow any other.
  - A · Jungle Exploration: the default forest floor.
  - B · River & Waterfall.
  - C · Canopy Vista: high treetops, open sky.
  - D · Moss Ruins: temples, caves, darkness.
  - E · Vine Run: fast traversal.
- **Layers per area** (in `stems/<area>/`, named `X_layer.ogg` where X is A–E):
  - Jungle, which leads:
    - `amb`, `amb2`, `amb3`, `amb4`: four takes of the ambience; play one per pass.
    - `water`
    - `wildlife`
    - `birds`: bird calls tuned to the harmony.
  - Music bed, which sits underneath and has no melody: `pad`, `keys`, `bass`, `woods`, `shaker`, `drums` and `marimba`, plus `kalimba` in B, C and E only.
  - All stems are pre-balanced: every layer at volume 1.0 is the intended mix.
- **Ring-out tails:** every non-bed layer has `_tail8` and `_tail16`. These are its reverb and echo decay when it stops at bar 9 or at the loop point.
- **Transitions** (`transitions/`): shower, gust, flock, far_cry, woodpecker_roll, fill_hands, fill_logs and rise. Each file's downbeat is at sample 128 000, one bar in.
- **One-shots** (`oneshots/`): dry creature and instrument sounds for placing in the world.

## 3. Required behaviour

### Timing (the most important part)

- Drive everything from the audio clock, never from frame time. Unity: `AudioSettings.dspTime`. Web Audio: `AudioContext.currentTime`. Otherwise use the engine's DSP or playback-position clock.
- Keep one conductor that knows the current area, when its current pass started, and therefore the current bar and 4-bar phrase.
- Plan each phrase boundary at least one bar plus about 0.7 s ahead. That leaves time to decode, and to start transition sounds that begin a bar early.
- Start and stop loops sample-accurately:
  - Unity: `PlayScheduled` / `SetScheduledEndTime`.
  - Web: `start(when, offset)` / `stop(when)`.
  - Godot 4.3+: consider `AudioStreamSynchronized` for a single area's layers and `AudioStreamInteractive` for bar-quantized area changes.
  - FMOD/Wwise: a 90 BPM timeline with quantized transitions.
- A layer that starts mid-pass must start at the offset that matches the pass position, so it lines up with the others.
- Load loops fully decoded (Unity: "Decompress On Load") so they loop without gaps. Never convert them to MP3, which adds padding and breaks the sample counts.

### Rules for change (from `docs/ENGINE.md`)

- **Layer changes:** only on 4-bar phrase boundaries. The harmony layers (`pad`, `keys`, `bass`) may only leave on 8-bar boundaries.
- **Area changes:** on 8- or 16-bar boundaries, and only after at least 8 bars in the current area. The new area starts at its sample 0.
- **Exits:**
  - A non-bed layer that stops at bar 9 or at the loop point is cut on the boundary (about 25 ms fade), and its `_tail8` or `_tail16` plays from that moment.
  - Beds (`amb*`, `water`, `wildlife`) crossfade over 3–5 s instead.
- **Ambience takes:** rotate `amb` → `amb2` → `amb3` → `amb4` on each new pass with a 4 s crossfade.
- **Transition sounds:** schedule one bar before the boundary they mark.

| Moment | Transition |
|---|---|
| Arriving at River & Waterfall | shower |
| Arriving at Canopy Vista | gust, plus flock if Openness is high |
| Arriving at Moss Ruins | far_cry |
| Arriving at Vine Run | woodpecker_roll, plus fill_hands when the music bed is up |
| Returning to Jungle Exploration | flock |
| The music bed drops away | gust or far_cry |
| The music bed builds | fill_logs or woodpecker_roll |

Play at most one transition every 8 bars.

- **Optional, if the engine allows negative gain:** when an area starts, you can play its layers' `_tail16` inverted alongside the first pass to remove the loop's own wrapped tail. Skip this if the engine can't invert a signal.

### Parameters: connect them to gameplay

Expose four smoothed values from 0 to 1 (about 1–2 s exponential smoothing), plus a music-bed amount:

| Parameter | Suggested gameplay source | Effect (port from `arrange` / `musicGain`) |
|---|---|---|
| **Intensity** | Player speed and momentum, vine swinging, chase or timed sequences, nearby threats | Raises the music bed's ceiling and brings in drums, shaker and marimba. High values lead to Vine Run. |
| **Openness** | Camera altitude or share of sky on screen, treetop zones, big vistas | Favours pad and keys over rhythm, and widens the reverb. High values lead to Canopy Vista. |
| **Water** | Distance to the nearest water volume or waterfall, swimming | Lifts the water layer and kalimba. High values lead to River & Waterfall. |
| **Mystery** | Ruin and cave zones, darkness, night | Thins the percussion, darkens the top end (a gentle low-pass) and lifts the night creatures. High values lead to Moss Ruins. |
| **Music bed** | Player setting or narrative moments | Scales all music-bed layers; 0 = pure jungle. |

- **Area choice:** take the largest of Water, Openness, Mystery and Intensity, compared against 0.45 for Jungle Exploration. Give the current area a 0.12 bonus so it doesn't flip back and forth.
- **Level data:** let designers author zones, such as rectangles in the level editor or Tiled object layers, with parameter values and a falloff distance. Blend overlapping zones, and let designers pin an area for a zone when they need to.
- **Breathing:** the music bed builds one level per phrase to a ceiling set by Intensity, holds, drops back to the jungle alone, then rebuilds with the other layer order. Port `dirStep` and the per-area `ORDER` table.

### One-shots in the world

- **Place emitters by biome:**
  - songbirds and bird calls in tree crowns
  - woodpeckers on trunks
  - tree frogs and frog croaks near water and in ruins at night
  - drips under leaves and in caves
  - cicadas in sunny areas
- **Randomise:** interval, volume (±3 dB) and pitch (±2%).
- **Position:** pan by screen x and attenuate by distance from the camera. Keep them sparse, because the ambience loops already carry the background.
- **Tie to visuals where it's cheap:** a bird flying off when its call plays, frogs blinking when they croak.

### Game events

- **Pause:** low-pass the mix to about 500 Hz and drop it about 9 dB. Resume in place, without restarting the loops.
- **Death and respawn:** fade the music bed to ambience, then bring it back on the next bar after respawn. Don't restart the music on every respawn.
- **Scene or level load:** keep the conductor alive across scenes that share the jungle. Preload the next level's likely areas.
- **Mixing:** give music, ambience (the jungle layers), SFX and UI separate buses with their own volume settings. Put a gentle limiter on the master.

### Musical gameplay events (port `player/events.js`)

The web player's `CanopyGame.event(name)` is the reference. `CanopyEvents.schedule(name, {bpm, now, origin, chord, key, chain})` is a pure function: it returns notes `{time, midi, dur, gain, voice}` and bus actions, so port it as-is and play the notes with your engine's synth, or with short pitched one-shots from `oneshots/` (kalimba, marimba, bell). Quantize against the conductor's clock (`origin` = the current area's start), never frame time. The chord is the current bar's label from `docs/ENGINE.md`.

| Game moment | Call | Music behaviour |
|---|---|---|
| Coin or fruit pickup | `collect` | next note of an in-chord ascending chain on the next 1/16; reset after 1.2 s without pickups |
| Jump | `jump` with `{sound: true}` | optional soft tick on the next 1/16; off by default |
| Checkpoint flag | `checkpoint` | rising chord arpeggio landing on the next beat |
| Secret found | `secret` | sparkle glissando on the next 1/16 |
| Damage | `hurt` | thud; low-pass and gain dip on the score for about 0.6 s |
| Player dies | `death` | falling figure; music fades and filters over one bar; ambience only until respawn |
| Respawn | `respawn` | music re-enters on the next bar; no restart |
| Level complete | `victory` | two-bar fanfare from the next bar line; duck the music bed under it |
| Boss arena | `boss` / `boss_end` | Full arrangement and intensity 1 from the next bar until the fight ends |
| Pause menu | `pause()` / `resume()` | low-pass to about 500 Hz and −9 dB; keep the clock and loops running |

Route event sounds to their own bus after the music ducks, so `hurt`, `death` and `victory` never duck themselves. Raise `bar`, `phrase` and `area` signals from the conductor, with the downbeat's DSP time, so visuals such as blinking platforms, pulsing lights and HUD beats can lock to the music.

### Performance

- A decoded 42.7 s stereo loop is about 8 MB at 16-bit or 16 MB as float.
- Keep only the current area, plus the one you're heading to, decoded. Decode the rest on demand, and unload areas that are not nearby.
- Start layers only when their gain is above zero, and stop them once they have been silent for a few seconds.

## 4. Deliverables

1. The conductor and audio system, ported from `player/index.html` to this engine, with the constants in one config: areas, layers, `ORDER`, gains and transitions.
2. Zone and parameter authoring for designers, with a short doc explaining how to tag a level.
3. A debug overlay, toggleable. It shows area, bar:beat, phrase, parameter values, active layers with their gains, and the next planned change. Debug keys force areas and parameters.
4. Unit tests for the timing math, the quantization rules and area selection with hysteresis.
5. A short README in the game repo explaining the system.

## 5. Acceptance checklist

- [ ] No clicks or gaps at loop points after 30 minutes of play. All layers stay aligned (check by soloing two layers).
- [ ] Layers change only on 4-bar boundaries, and harmony layers leave only on 8-bar boundaries.
- [ ] Area changes happen only on 8- or 16-bar boundaries, and never faster than every 8 bars.
- [ ] Exiting layers ring out through their tails, with no abrupt reverb cut-offs. Beds crossfade smoothly.
- [ ] Transition sounds land exactly on the downbeat they mark.
- [ ] The ambience take rotates every pass.
- [ ] Pause and resume, death and respawn, and level loads behave as described.
- [ ] No clipping on the master in the loudest configuration.
- [ ] Memory stays within the budget you set, and the frame rate is unaffected.
- [ ] With the music bed at 0, the game plays with only the jungle.

Ask before changing gameplay code beyond adding zone data and the hooks the audio needs.
