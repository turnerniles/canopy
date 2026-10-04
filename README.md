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

### Connect gameplay

While a fusion mode is loaded, a same-origin game host can supply normalized telemetry:

```js
window.CanopyGame.setState({
  intensity: 0.65,   // action or danger, 0–1
  movement: 0.8,     // speed / movement, 0–1; opens the filter
  progress: 0.5,     // level progress, 0–1
  performance: 0.7,  // combo / performance, 0–1
});
window.CanopyGame.getState();
```

Values are clamped to 0–1 and non-finite inputs are rejected. Partial updates preserve the other values. A same-origin parent can call `iframe.contentWindow.CanopyGame.setState(...)`. Audio still requires the player's Play gesture. The layer board shows the applied arrangement; gameplay controls show the requested energy, which lands on the next available phrase. No external game integration is installed automatically.

The implementation is in `player/fusion.js` and `player/fusion-worker.js`. These modes are browser scores; the committed downloadable stems and demos remain the original jungle score.

Validate all generated stems, loop seams, release tails, mix headroom and gameplay builds with:

```sh
node player/fusion.test.cjs
node player/space.test.cjs
```

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
