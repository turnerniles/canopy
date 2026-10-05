# Canopy

An adaptive jungle soundscape for a side-view pixel-art adventure. Each of five areas has its own synthesized jungle: insects, birds, frogs, water and wind. An optional soft music bed sits underneath and builds up and falls away as the player moves. There is no melody; the jungle leads.

Every sound here is generated from code. There are no recordings, samples or existing music.

## What's in the repo

| Folder | Contents |
|---|---|
| `player/` | The interactive web player: environment sliders, a music-bed control and a live pixel-art scene |
| `stems/` | Engine-ready loops per area: 48 kHz Ogg Vorbis, each exactly 2 048 000 samples, with ring-out tails |
| `transitions/` | Jungle transition sounds (shower, gust, flock, far cry, woodpecker roll) and percussive fills |
| `oneshots/` | Dry creature and instrument one-shots for random emitters |
| `demos/` | Two 6½-minute walkthroughs: the jungle over a soft bed, and pure jungle |
| `docs/ENGINE.md` | Tempo, key, harmony per area, the layer list, and the rules for wiring it into a game engine |
| `docs/INTEGRATION_PROMPT.md` | A ready-to-paste brief for an AI assistant integrating Canopy into a 2D jungle platformer |
| `src/` | The Python that synthesizes, arranges, mixes and encodes everything |

## Listen

The quickest way in is `demos/canopy_pure_jungle.mp3`, then `demos/canopy_jungle_with_soft_bed.mp3`. Both walk through all five areas.

## Run the player

The player fetches its audio, so serve the folder rather than opening the file directly:

```sh
cd player
python3 -m http.server 8000
```

Open http://localhost:8000 and press Play. The audio is AAC, which Chrome, Edge, Safari and Firefox on Windows or macOS all decode. It works best on a desktop with headphones.

| Control | What it does |
|---|---|
| Intensity, Openness, Water, Mystery | Choose the area and shape which layers play |
| Music bed | Runs from 0 (pure jungle) up to the full soft bed |
| Area | Follow the environment, or pin one area |
| Music arrangement | Breathing (builds and drops on its own), Sparse, Medium or Full |
| Auto-explore | Moves the sliders like a player walking between places |
| auto / on / off on each layer | Forces that layer; the change lands on the next bar |

## Space soundtrack

Choose **Space** in the player's Soundtrack menu, or open `http://localhost:8000/?soundscape=space`, then press Play. Switching soundtracks starts a fresh session.

Space synthesizes its own audio in the browser: stellar drones, ion washes, radio signals, beacon chimes, nebula pads, glass keys, sub bass and electronic pulses. A starfield and ringed planet accompany five destinations: Orbital Drift, Ion Nebula, Deep Field, Derelict Station and Warp Corridor. The existing arrangement, layer, intensity and auto-explore controls still work; the Water slider becomes Nebula. Turn Music bed to zero for space ambience alone.

Each destination has its own 16-bar progression on Canopy's grammar (bar 1 on D, bars 8 and 16 on A), so it bridges cleanly to and from the jungle: D major and Lydian colours in Orbital Drift and Ion Nebula, slow two-bar harmony in Deep Field, D minor in Derelict Station and a driving progression in Warp Corridor. The pad re-voices every chord with smooth voice-leading, the sub bass follows the roots, the sequencer arpeggiates the bar's chord, and a gliding theremin lead plays fragments of Canopy's A–D–E–A motif transposed to each chord.

The synthesizer lives in `player/space.js`, with deterministic, sample-aligned loops and release tails at 24 kHz. It requires no generated audio files or additional dependencies. The committed engine stems and downloadable demos remain the jungle score.

## Progressive fusion modes

The Soundtrack menu also includes six original acoustic/electronic scores:

| Mode | Tempo | Palette |
|---|---|---|
| Jungle Canopy Level (`?soundscape=canopy`) | 120 BPM | Slap bass, congas, steel drums, disco strings, house kit and synth arpeggio |
| Mine Cart Level (`?soundscape=mine`) | 128 BPM | Mechanical ambience, crushed metal, funk guitar, acid bass and synthesized robotic vowels |
| Underwater Level (`?soundscape=underwater`) | 105 BPM | Glassy pads, breathy flute, deep marimbas, cinematic strings and pumping analog bass |
| Orbital Rainforest (`?soundscape=orbital`) | 112 BPM | Alien calls, crystal rain, hollow wood, liquid bass, cosmic strings and metallic replies |
| Crystal Caverns (`?soundscape=crystal`) | 100 BPM | Bowed glass, mineral drips, stone mallets, subterranean bass and a quartz sequencer |
| Clockwork Greenhouse (`?soundscape=greenhouse`) | 125 BPM | Reed harmonium, clock mechanisms, music-box tones, swung funk, porcelain bells and gear arpeggios |

These modes use original procedural motifs and instrument approximations, without recordings or quoted melodies. The robotic hook is formant-synthesized vowel sound, not intelligible lyrics or a recorded vocalist. Each mode has five destinations and its own scene, and is rendered in a background worker without additional packages.

Use **Exploring / quiet**, **Action / platforming**, or **Danger / climax** to preview the build. In **Adaptive** arrangement, effective energy is the maximum of Intensity, Progress × 0.65 and Performance × 0.85. Below 34% it plays atmosphere, flute and a distant kick; from 34% the rhythm and bass enter; at 74% the strings and arpeggio or robotic hook join. Sparse, Medium and Full pin these three arrangements. Layer overrides still work. Music bed controls all accompaniment; zero leaves the atmosphere and flute.

**Movement** opens a smoothed global low-pass filter from 650 Hz to roughly 16.7 kHz; Mystery lowers its ceiling. Electronic layers duck in time with the kick, with stronger pumping at higher energy. A moving all-pass filter colours the strings and machine stabs.

All stems inside a mode share an exact 16-bar grid and four-bar harmonic progression. Canopy uses D major colours; Mine, Underwater and Orbital Rainforest use D minor; Crystal Caverns uses E minor; Clockwork Greenhouse uses G major. Their 28 kHz render rate gives integer samples per beat at every configured tempo. Layer changes happen on four-bar phrases; destinations change on eight-bar boundaries with release tails and ambience crossfades. Switching soundtrack reloads the player; it does not crossfade between different tempos. Browser playback resamples to the audio device rate.

Orbital Rainforest has a specific transformation: its scattered creature calls fade down as a rhythmic layer using the same synthesized creature voice fades up. At climax, metallic replies and cosmic strings join. Its five destinations are Spore Landing, Lunar Falls, Ringworld Canopy, Sleeping Monolith and Meteor Bloom, beneath a ringed planet and luminous forest.

Crystal Caverns starts with slowly ringing minerals and water drops before building its deep techno pulse and quartz sequence. Its destinations are Quartz Entrance, Dripstone Pool, Amethyst Vault, Obsidian Chamber and Prismatic Heart, with glowing crystal formations and reflecting pools.

Clockwork Greenhouse combines scattered mechanisms with swung percussion and a music box, then adds muted guitar, plucked bass, reed stabs and interlocking arpeggios. Seedling Atrium, Irrigation Works, Glass Conservatory, Overgrown Orrery and Clocktower Bloom share a glass-roofed conservatory of turning gears and plants.

The new modes use the same gameplay controls and API as the other fusion scores. Their environmental sliders are Alien rain, Seepage and Irrigation respectively. Each has its own reverberation, melodic motifs and percussion patterns.

The implementation is in `player/fusion.js` and `player/fusion-worker.js`. These modes are browser scores; the committed downloadable stems and demos remain the original jungle score.

## Connect gameplay

Every soundtrack (Jungle, Space and the fusion modes) exposes the same API to a same-origin game host, or to a parent page through `iframe.contentWindow.CanopyGame`. Audio still needs the player's Play gesture; calls made before then update state only.

```js
const C = window.CanopyGame;
C.setState({
  intensity: 0.65,   // action or danger; drives the Intensity slider
  movement: 0.8,     // player speed; opens the master low-pass in every mode
  progress: 0.5,     // level progress; raises effective intensity (× 0.65)
  performance: 0.7,  // combo / momentum; raises effective intensity (× 0.85)
  openness: 0.3, water: 0.1, mystery: 0,   // drive the environment sliders and the area choice
  area: 'auto',      // 'A'..'E' pins an area, 'auto' follows the environment
});
C.getState();        // values plus area, currentArea, stage, energy, bar, beat, bpm, boss, dead, paused
C.event('collect');  // beat-quantized, in-key one-shot (see below)
C.pause(); C.resume();   // pause-menu treatment: low-pass to ~500 Hz and −9 dB; the clock keeps running
const off = C.on('bar', ({bar, phrase, area, time}) => {}); // also 'phrase' and 'area'; returns an unsubscribe
```

Numbers are clamped to 0–1; non-finite values, unknown areas, events and listener names throw `TypeError`. Partial updates keep the other values. Effective intensity is the maximum of Intensity, Progress × 0.65 and Performance × 0.85; it sets the arrangement, while the environment values choose the area. In Jungle and Space, Movement defaults to 100% so the mix is unchanged until a game drives it; there it sweeps the filter from 1.2 kHz up to the Mystery ceiling. Clock events fire as each bar starts; `time` is the downbeat on the `AudioContext` clock, `bar` is 1–16 and `phrase` 1–4.

| Event | Sound | Lands on |
|---|---|---|
| `collect` | next note of an ascending chain built from the current bar's chord; resets after 1.2 s idle | next 1/16 |
| `jump` | very soft tick, silent unless `{sound: true}` | next 1/16 |
| `checkpoint` | rising arpeggio of the current chord | top note on the next beat |
| `secret` | sparkle glissando through the chord | next 1/16 |
| `hurt` | muted thud; the score dips and low-passes for about 0.6 s | next 1/16 |
| `death` | falling minor figure while the music fades and filters down over one bar; ambience only until `respawn` | next 1/16 |
| `respawn` | music returns from the next bar | next beat |
| `victory` | two-bar fanfare in the soundtrack's key; the music bed ducks under it | next bar line |
| `boss` / `boss_end` | Full / Climax arrangement at intensity 1 from the next bar, until `boss_end` | next beat |

Events are synthesized at runtime by `player/events.js` (no audio files): kalimba in Jungle, an FM glass bell in Space, steel drum or mallet in the fusion modes, plus brass, thud and noise-swell voices. `event()` returns `{name, time}`, where `time` is the quantized `AudioContext` time. The Gameplay panel has a button for each event so you can audition them. In Space and the fusion modes, area changes and big arrangement changes are marked by a synthesized riser that starts one bar before the boundary and lands on it.

Validate all generated stems, loop seams, release tails, mix headroom, harmony, gameplay builds and event scheduling with:

```sh
node player/fusion.test.cjs
node player/space.test.cjs
node player/events.test.cjs
```

## KAPOK in the player

Choose **KAPOK (Oldfield-style suite)** in the Soundtrack menu, or open `http://localhost:8000/kapok.html`. The page plays the 10-minute suite against a live score view:
- a pixel-art climb up the kapok tree, from the roots to the stars
- a movement timeline you can click to jump
- a "now entering" ticker and every instrument currently sounding
- the full credits for all 234 instruments
- a heat strip per instrument family
- the optional announcer for the Procession, scheduled on the audio clock and ducking the music

`CanopyGame.event('collect' | 'checkpoint' | 'hurt' | 'secret' | 'victory')` lands on Kapok's own beat grid and chords, including the 15/8, 12/8 and 3/4 sections. `CanopyGame.on('bar' | 'movement', fn)` follows the piece.

The page loads `audio/kapok.m4a`. If that binary isn't in the checkout, it falls back to the same bytes stored as base64 chunks in `audio/kapok_b64/`. Regenerate both with `python3 -m tracks.kapok_web ../out/kapok`.

## Microgame Rush (WarioWare-style)

Choose **Microgame Rush** in the Soundtrack menu, or open `microgame.html`. It's a playable run made of bar-exact cues:
- interlude → microgame (one of 8 styles) → win/lose jingle
- a speed-up every 4 games, across 4 tempo tiers (120–180 BPM)
- a boss at game 12, and 4 lives

The demo's five one-button microgames are Tap!, Tap ×3!, Don't tap!, Wait for green and Stop on the star. The page has an auto-play bot, English and Spanish commands, and a cue board for auditioning every cue at every speed.

Render the engine pack (OGG per cue plus `cues.json`) with `cd src && python3 -m tracks.microgame --web`. The run grammar and integration notes are in `docs/MICROGAME.md`.

## Canopy Orchestra: build your own soundtrack

`src/orchestra/` turns Canopy from one soundtrack into a builder. See `docs/CRITIQUE.md` for why.

| Module | What it does |
|---|---|
| `core.py` + `inst_*.py` | 100+ synthesized instrument presets behind one contract, auto-calibrated for loudness. List them with `python3 -m orchestra.check_instruments`. |
| `theory.py` | Chord symbols (`Bbmaj7#11`, `A7sus4b9`, slash chords), voice-leading, scales (modes, pentatonics, pelog, slendro), motif transforms |
| `mix.py` | Sessions, parts, odd meters (`Meter([15]*8, unit=8)`), tempo maps, rooms from booth to cosmos, tape echo, automation, mastering |
| `world.py` | A JSON world spec becomes engine-ready loops, tails, in-key stingers and `meta.json` |
| `src/worlds/*.json` | Example worlds: a jungle temple and an orbital station, sharing Canopy's cadence grammar so they can follow each other |
| `src/tracks/kapok.py` | **KAPOK**, an original Oldfield-style suite that plays the whole library |

```sh
cd src
python3 -m orchestra.check_instruments                 # verify every preset
python3 -m orchestra.check_instruments creature --wav ../out/creatures.wav
python3 -m orchestra.selftest                          # fast regression checks
python3 -m orchestra.world worlds/jungle_temple.json --preview      # -> src/out/worlds/jungle_temple/
python3 -m orchestra.world worlds/orbital_station.json --preview
python3 -m orchestra.walkthrough out/worlds/jungle_temple:A,D,E out/worlds/orbital_station:A,D,E --passes 1,3 -o walk.wav
python3 -m tracks.kapok --out ../out/kapok             # the 9:22 suite, about 2 minutes on 2 cores
python3 -m tracks.kapok_mc ../out/kapok MC_DIR         # optional: overlay spoken introductions (mc_01.wav ...)
```

A built world folder holds `<area>_<layer>.ogg` loops (exact length, seamless), `_tail8` / `_tail16` ring-outs, `stingers/` (in-key collect notes, checkpoint, victory, death, `to_<area>` transitions) and `meta.json`. That file follows the `player/meta.json` schema plus a `cues` block: tiers per layer, and the chord tones of every bar, so a game can pitch its own pickups to the music.

### A world in a few lines

```json
{"id": "my_level", "bpm": 96, "key": "D", "mode": "lydian",
 "areas": {"A": {"name": "Sunlit Ruins", "room": "canopy",
   "progression": ["Dmaj9","E/D","Bm11","Gmaj9", "...16 chords..."],
   "layers": {
     "pad":  {"inst": "bowed.section_warm", "pattern": "pad", "tier": 1, "gain_db": -10},
     "bass": {"inst": "bass.fretless", "pattern": "bass", "rhythm": "x..x..5.x...a...", "tier": 2},
     "cell": {"inst": "mallet.balafon", "pattern": "ostinato", "degrees": [1,5,2,5,3,5,2], "rhythm": "xxx.xx.x", "tier": 2},
     "frogs":{"insts": ["creature.tree_frog_choir"], "pattern": "scatter", "density": 1.5, "tier": 1}}}}}
```

Pattern generators: `pad`, `comp`, `bass`, `arp`, `ostinato`, `motif`, `drums`, `scatter` and `drone`. Tiers 1–3 map to explore, action and climax. Every progression must start on the home chord and rest on the dominant at bars 8 and 16, so areas and worlds can follow one another.

## The areas

| Area | The jungle | Music bed |
|---|---|---|
| A · Jungle Exploration | Insects, distant birds, frogs, leaves; cicada swells, songbirds, woodpeckers | Pad, e-piano, bass, wood knocks, shaker, hand drums, marimba |
| B · River & Waterfall | A brook, a waterfall tuned to the chords, tuned drips, tree frogs | Shimmering pad, kalimba ripples, e-piano arpeggios, soft frame drum |
| C · Canopy Vista | Wind over the canopy, distant raptor cries, a far waterfall | Wide pad, long bass notes, e-piano swells, felt-tom swells |
| D · Moss Ruins | Crickets, frogs, wind resonating in stone, cave drips | A D–A drone with shifting colours, heartbeat toms, seed rattles |
| E · Vine Run | Busy birdsong, cicadas, woodpeckers | Driving bass, 16th-note marimba, hand drums, shaker |

It runs at 90 BPM in 4/4, with 16-bar loops (42.667 s). The key is D, Lydian-tinged, with D minor colours in the ruins. Layers change on 4-bar phrases and areas change on 8- or 16-bar boundaries. See `docs/ENGINE.md` for the full rules.

## Regenerate the audio

Needs Python 3.10+ and `ffmpeg` built with libvorbis and AAC.

```sh
pip install -r requirements.txt
cd src
python render.py              # every layer of the full score, with ring-out tails (about 10 minutes)
python wild.py                # jungle layers, alternate ambience takes, jungle transitions
python stingers.py            # percussive transitions
python mix.py                 # loudness balance of the full score
python jungle_mix.py          # jungle-forward balance and the two walkthrough mixes
python encode_web_jungle.py   # player audio -> src/out/webj
python pack.py                # engine pack -> src/out/pack
```

Output lands in `src/out/`, which git ignores. The committed `player/audio`, `stems/`, `transitions/` and `oneshots/` are copies of those outputs. `render.py` also renders the melody layers of the original score; the jungle version leaves them out.
