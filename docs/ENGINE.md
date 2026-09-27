# Canopy — adaptive jungle soundscape

An original, fully synthesized adaptive score for a side-view jungle adventure. The jungle is the lead voice. A soft music bed with no melody sits underneath and builds up and falls away. Every sound in this pack was generated from code. There are no recordings, samples or existing music.

## The grid

| | |
|---|---|
| Sample rate | 48 kHz, stereo |
| Tempo | 90 BPM, 4/4 |
| One beat | 32 000 samples (0.667 s) |
| One bar | 128 000 samples (2.667 s) |
| One loop | 16 bars = 2 048 000 samples (42.667 s) |
| Phrase | 4 bars = 512 000 samples |

Every loop file in `stems/` is exactly 2 048 000 samples and loops seamlessly. Reverb and echo tails are rendered circularly, so the end of the loop flows into its start. All layers of an area start on the same sample and stay phase-locked.

## Key and harmony

The home key is D major with a Lydian colour: G# appears often, most clearly in the E/D chord. The Moss Ruins shift to D minor colours over a sustained D–A drone. Every area obeys the same two rules, so any area can follow any other:

- bar 1 is always a D-rooted chord (Dmaj9, or Dm9 in the ruins)
- bars 8 and 16 always rest on an A-rooted chord (Asus2, A, A7, A9sus4)

The signature colour is the borrowed ♭VI and ♭VII (Bbmaj9, Cmaj9) near the end of the phrase, plus major IV turning to minor iv (Gmaj9 → Gm6/9) in the vista.

| Area | Bars 1–16 |
|---|---|
| A · Jungle Exploration | Dmaj9 · E/D · Bm11 · Gmaj9 · D/F# · Em9 · Gmaj9 · Asus2→A · Gmaj9 · F#m11 · Em9 · A13sus · Bbmaj9 · Cmaj9 · Em11 · A9sus4 |
| B · River & Waterfall | Dmaj9 ×2 · Bm9 ×2 · Gmaj7#11 ×2 · Asus2 · A · F#m9 ×2 · Gmaj9 ×2 · Bbmaj7#11 ×2 · Asus4 · A |
| C · Canopy Vista | Dmaj9 · Dmaj9/C# · Bm11 · Bbmaj7#11 · Gmaj9 · Gm6/9 · Em11 · A9sus4→A · Gmaj9 · A/G · F#m11 · Bm9 · Em9 · Gm6/9 · Bbmaj9 · A9sus4→A |
| D · Moss Ruins | Dm9 ×2 · Bbmaj7#11 ×2 · Gm6/9 ×2 · A7sus4♭9 · A7 · Fmaj7#11 ×2 · Ebmaj7#11 ×2 · Bbmaj9 ×2 · Asus4 · A (over a D–A drone) |
| E · Vine Run | Dmaj9 · E/D · Dmaj9 · E/D · Bm11 ×2 · Gmaj9 · Asus4→A · Gmaj9 · A/G · F#m11 · Bm9 · Em9 · F#m7 · Gmaj9 · A9sus4 |

## Layers

Each area folder holds these files. `X` is the area letter.

**Jungle (the lead)**

| File | What it is | Behaviour |
|---|---|---|
| `X_amb`, `X_amb2`, `X_amb3`, `X_amb4` | The area's ambience: insects, distant birds, frogs, leaves, wind | Four takes of the same recipe. Play one per pass and rotate on the loop point with a 4 s crossfade, and the forest won't repeat for about 2:50. |
| `X_water` | Brook bubbles, a waterfall whose spray is tuned to the chords, tuned drips | Bed: crossfade 3–5 s |
| `X_wildlife` | Cicada swells, songbird phrases, woodpeckers, tree frogs, drips off leaves | Bed: crossfade 3–5 s |
| `X_birds` | Bird calls pitched to the harmony, placed at phrase ends | Tails available |

**Music bed (underneath, no melody)**

`X_pad`, `X_keys` (warm tine e-piano), `X_bass`, `X_woods` (tuned log-drum knocks), `X_shaker`, `X_drums` (soft hand drums, felt kick, frame drum, toms), `X_marimba`. River, Vista and Vine Run also have `X_kalimba`: ripples and plinks that are texture, not a tune.

All stems are pre-balanced. Every layer at full volume is the intended jungle-forward mix.

## Adaptive rules (what the web player does)

- **When layers change:** only on 4-bar phrase boundaries, at samples 0, 512 000, 1 024 000 and 1 536 000 of the loop.
- **Harmony layers:** `pad`, `keys` and `bass` only leave on 8-bar boundaries, at sample 0 or 1 024 000.
- **Area changes:** on 8- or 16-bar boundaries. Start the new area's loops at their sample 0.
- **Ring-outs:** when a non-bed layer stops at bar 9 (sample 1 024 000), cut it and play `X_layer_tail8` from that moment. When it stops at the loop point, play `X_layer_tail16`. Its reverb and echoes then decay naturally.
- **Cleaner area entry (optional):** each loop's first seconds contain its own wrapped tail from bar 16. When an area starts from silence, you can play that layer's `_tail16` inverted (gain −1) alongside the first pass to remove it.
- **Transitions** (`transitions/`): each file's downbeat is at sample 128 000, one bar in. Schedule it one bar before the boundary it marks.

| Transition | Use |
|---|---|
| `shower` | Arriving at the river |
| `gust` | Arriving at a vista, or when the music drops away |
| `flock` | Birds taking off: returning home, or at open vistas |
| `far_cry` | A distant raptor cry: entering the ruins, or when the music drops away |
| `woodpecker_roll` | Into the vine run, or when the music builds |
| `fill_hands`, `fill_logs`, `rise` | Percussive fills, used when the music bed is up |

## Parameters

| Parameter | Effect |
|---|---|
| **Intensity** | Raises the music bed's ceiling. Brings in hand drums, shaker and marimba, and makes the wildlife busier. High intensity leads to the Vine Run. |
| **Openness** | Favours pad and e-piano over rhythm, and adds room reverb. High openness leads to the Canopy Vista. |
| **Water** | Lifts the water layer and kalimba ripples. High water leads to the River & Waterfall. |
| **Mystery** | Thins the percussion, darkens the top end and brings up night creatures. High mystery leads to the Moss Ruins. |
| **Music bed** | Scales the whole music bed from 0 (pure jungle) to 1. |

The music bed "breathes": it builds one level per phrase to a ceiling set by Intensity, holds, drops back to only the jungle, then rebuilds with a different combination of layers.

## One-shots

`oneshots/` holds dry creature and instrument one-shots for random emitters placed in the world:

- cicada swells, songbirds, woodpeckers, tree frogs, frog croaks, bird calls
- tuned drips, bubbles, drips off leaves
- log drums, hand drums, toms, shakers
- kalimba, marimba and bell notes in the key

## Source

`source/` (`src/` in the repository) is the complete Python (NumPy/SciPy) code that synthesizes, arranges and renders everything, so the pack can be regenerated. It covers the instruments, score, arranger, jungle layers, transitions, conductor and encoders. Render order:

1. `render.py`
2. `wild.py`
3. `stingers.py`
4. `jungle_mix.py`
5. `encode_web_jungle.py`
6. `pack.py`

Set `KEEP_WAV=1` to keep 24-bit WAV masters.
