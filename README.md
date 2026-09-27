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
