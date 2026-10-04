# Canopy as a soundtrack *builder* — review and roadmap

Reviewed for its intended use: building an adaptive soundtrack for a 2D pixel-art platformer that starts in a jungle and moves into space.

## What is already excellent

- **The adaptive grammar is professional grade.** Phase-locked 16-bar loops, sample-exact grids, circular rendering so reverb and echo wrap seamlessly, `_tail8` / `_tail16` ring-outs, phantom-tail cancellation on entry, layer changes on 4-bar phrases, harmony that only leaves on 8-bar lines, and a shared cadence rule (bar 1 on D, bars 8 and 16 on A) that lets any area follow any other. That is the horizontal-plus-vertical model FMOD and Wwise users build by hand.
- **Environment parameters, not tracks.** Intensity, Openness, Water and Mystery choose the area and shape the layers. The "breathing" arrangement keeps a long session from feeling looped.
- **Careful DSP.** The circular IIR filters, the frequency-domain feedback echo and the multiband room IRs are the right tools for loop-exact stems.
- **Documentation aimed at the integrator.** `ENGINE.md` and `INTEGRATION_PROMPT.md` make it portable.

## Where it falls short as a builder

| # | Problem | Why it matters for a platformer |
|---|---|---|
| 1 | **It is one hand-written soundtrack, not a tool.** All content lives in code: `sec_A`…`sec_E` (about 100 lines of DSP calls each) and the `id==='mine'?…` chains in `fusion.js`. There is no data format for a world, an area or a layer. | Every new level means writing synthesis code. A 30-level game cannot be scored that way. |
| 2 | **Two engines of very different quality.** The Python jungle engine is rich. The browser scores (six fusion modes and Space) are per-sample JS: about 20 voices in one `if` chain, a 4-chord loop repeated four times, and fixed patterns (a kick on every beat). There is no humanisation and no per-stem space. Mine, Underwater and Orbital share the identical chord table. | The modes sound alike after a minute, which is exactly when a player is still in the level. |
| 3 | **Space, the second half of the game, is the weakest part.** Each area holds **one chord for all 42 seconds** (`Array(16).fill(chord)`), uses four timbres and has no transitions (stingers are disabled for synthesized modes). | Half the game would be scored by a drone. |
| 4 | **A small, single-use instrument palette.** About 15 instruments live in `synth.py`, written for one piece. There is no registry, no loudness calibration and no shared velocity-to-timbre behaviour. | You cannot ask for "kora here, talking drum there". Layered, many-instrument writing (the Oldfield way) is impossible. |
| 5 | **A rigid grid.** 90 BPM, 4/4 and 16 bars are module-level constants (`SPB`, `BAR`, `N`). There are no odd meters, tempo maps or per-world tempo. Changing soundtrack reloads the page. | Boss fights, chases and space levels want their own tempo and meter. The jungle-to-space handoff has no musical bridge. |
| 6 | **No game-feel layer.** There are no beat-quantised stingers for collect, checkpoint, hurt, death, victory, boss or secret, no pause treatment, no in-key collectible chains and no bar/beat callbacks. `CanopyGame` exists only in the fusion modes. | These cues are most of what makes platformer audio feel *good*: Mario-style rising coin chains and DKC-style ducking. |
| 7 | **No melody by design.** "The jungle leads" is lovely ambience, but platformers live on themes. The original melody layers exist in `score.py` and are thrown away. | Players remember levels by their tune. The A–D–E–A Canopy motif could be the game's spine from jungle to space. |
| 8 | **The pipeline is brittle.** Six scripts run in a fixed order (about 10 minutes), with no "build one level" command and no Python tests. The browser and Python layer tables (`ORDER`, area texts) are duplicated and can drift apart. Browser rendering runs at 24–28 kHz. | Iteration is slow, and errors surface late. Browser modes sound dull at the top end. |

## What this change adds

1. **Canopy Orchestra (`src/orchestra/`), with 234 synthesized instrument presets in 17 families.**
   - **The families:** mallets, bells, gamelan and metal, drums, small percussion, plucked strings, guitars, basses, bowed strings, keyboards, organs, flutes, reeds, brass, wordless voices, *creatures as pitched instruments* and space synths.
   - **One contract:** every preset uses `fn(f, dur, vel, rng, sr)`, changes timbre with velocity, is deterministic, and is auto-calibrated to comparable loudness.
   - **Verification:** `check_instruments.py` checks pitch (±35 cents), tails, clipping and render speed for every preset. Current result: 0 failures.
2. **A score format with odd meters and tempo maps (`mix.py`).** It provides Sessions, Parts and `Meter([15]*8, unit=8)`, plus per-part EQ and sends to nine named rooms from booth to cosmos. It adds tape echo, level automation, and octave-folding into each instrument's range.
   - **Mastering:** bus compression, tape saturation, LUFS normalisation and a lookahead limiter.
   - **Long pieces:** `render_chunked` keeps memory bounded, and it matches a one-pass render to −135 dB.
3. **Music theory (`theory.py`).** A chord-symbol parser (`Bbmaj7#11`, `A7sus4b9`, `Gm6/9`, slash chords), voice-leading, scales (modes, pentatonics, pelog, slendro) and motif transforms.
4. **A data-driven world builder (`world.py` + `src/worlds/*.json`).**
   - **The spec:** one JSON file per world: tempo, key, a 16-bar progression per area, and layers made of an instrument plus a pattern generator (pad, comp, bass, arp, Oldfield ostinato, leitmotif, drums, creature scatter, drone), each with a tier and sends.
   - **What it renders:** Canopy-style circular loops (exact length, pre-roll wrapped so early pickups don't click), `_tail8`/`_tail16`, in-key stingers (collect notes, checkpoint, victory, death, `to_<area>`), one shared auto-gain across areas, and a `meta.json` with tiers and per-bar chord tones.
   - **Validation:** it checks the cadence grammar and warns about tails that are too short.
   - **Shipped worlds:** a jungle temple (4 areas, 40 instruments) and an orbital station (4 areas, 24 instruments). Both use the same grammar and leitmotif, so the jungle can hand over to space on any 8-bar line.
   - **Auditioning:** `walkthrough.py` plays built worlds the way the ENGINE.md rules would.
5. **The browser player.** One `CanopyGame` API covers every soundtrack, including `on('bar'|'phrase'|'area')` clock callbacks. It adds beat-quantised game events (`events.js`):
   - an in-key rising collect chain
   - checkpoint, hurt duck, death and respawn
   - a victory fanfare on the next bar line
   - secret and boss
   - a pause treatment

   Space now has real 16-bar progressions with 9 voices, and synthesized modes get transition risers.
6. **KAPOK (`src/tracks/kapok.py`).** An original Oldfield-style suite, 9:22 long, that plays all 234 instruments. It doubles as the library's showcase and its regression test.
7. **Tests.** `python3 -m orchestra.selftest` covers parsing, determinism, chunked versus full rendering, seams, tails and grammar. `player/events.test.cjs` is new, and `space.test.cjs` is extended.

## Recommended next steps

- Port the five hand-written jungle areas onto `world.py` specs, so the shipped score becomes data too. Keep `render.py` as the reference until A/B listening approves the port.
- Let the web player load a built world (`?world=jungle_temple`) straight from `out/worlds/<id>/meta.json`.
- Add a Godot exporter that writes an `AudioStreamInteractive` / `AudioStreamSynchronized` resource from `meta.json`.
- Add CI that runs `orchestra.selftest`, `check_instruments` and the Node tests on every PR.
- Listen to everything and tune it by ear. All of this work was verified numerically: spectra, pitch, envelopes, loudness and seams. The likeliest by-ear fixes are the choir and chant voices, the cymbal brightness, and the event-stinger levels in the browser.
