# Glimmer Legends: a Rayman Legends–inspired soundpack

Pick **Glimmer Legends** in the Soundtrack menu, or open `player/legends.html`. It's all original music on Canopy Orchestra's synthesized instruments, with no samples and no quoted melodies. It borrows what makes that kind of score work: a whimsical band per world, collectibles that play the tune, and musical levels whose gameplay is choreographed to the music.

## Worlds

Each world is an 8-bar level theme in **three phase-locked layers** that stack: 1 explore, 2 action, 3 chase. Layer changes land on the next bar line.

| World | Feel | Key, meter, tempo | Explore | Action | Chase |
|---|---|---|---|---|---|
| Briar Keep | fairy-tale castle | G major, 6/8, 120 | harp, strings, celesta | pizzicato, bassoon, frame drum | whistling, recorder, glockenspiel, teensies |
| Toadstool Bayou | swamp jug band | F major, swing, 108 | banjo, jaw harp, tree frogs | tuba, washboard, accordion, brushes | kazoo, low kazoo, slide whistles, bullfrog |
| Fiesta Calavera | day-of-the-dead | D minor, 2-beat, 132 | nylon guitar, guitarrón, marimba | accordion, maracas, claves, conga | trumpets in thirds, violin, cheer |
| Lagoon Agents | undersea spy surf | E minor, 120 | vibes, walking bass, ride and brushes | surf guitar, bongos, Farfisa | flute, muted trumpet and trombone stabs |
| Thunder Olympus | mythic temples | D minor, 100 | harp, choir "ooh", low strings | timpani, bass drum, horns, pizzicato | trumpet fanfare, choir "aah", cymbals, glockenspiel |

Every world also has five **stingers** in its own key: `checkpoint`, `freed` (the little creatures cheer, with a harp glissando), `king` (glims turn golden), `death` (slide whistle, bonk, a sad kazoo "wah-wah") and `complete` (fanfare, crash and cheer).

## Glims

The collectible chime exists at every semitone from C5 to C7, in a normal version and a golden **king** version. A game plays each pickup as the *next tone of the current bar's chord*: `legends.json` lists `chord_tones` per bar for every world. A row of pickups then plays a melody that fits the harmony. The page quantises pickups to a 16th note, and a chain resets after 1.2 s without a pickup.

## The musical level: Kazoo Rumble

Kazoo Rumble is 48 bars at 132 BPM (about 1:32): intro, verse ×2, break, chorus ×2, bridge, finale. It's an A-minor gypsy polka with kazoo, accordion, banjo, tuba and brass.

The **action lane** is part of the score (8 eighth-note slots per bar):

| Action | What you hear on that exact beat |
|---|---|
| `J` jump | brass "ba!" stab and cymbal |
| `P` punch | snare, clap and trombone hit |
| `S` slide | a cymbal swell landing on the beat, and a slide whistle down |
| `G` glim | nothing in the track: the game plays the glim, whose note (in the beatmap) is the next chord tone, so the pickups play a melody |

The lane is exported as a beatmap (`level.beatmap` in `legends.json`), 136 events like `{"t": 7.273, "bar": 4, "slot": 0, "action": "J", "chord": "Am"}`. A level designer places a crate at every `J`, an enemy at every `P`, a low branch at every `S` and a glim at every `G`, at `x = t × run speed`. The page does exactly that. **Watch it** auto-plays. **Play it yourself** judges your timing: within 65 ms is Perfect, within 140 ms is Good.

## Files

`cd src && python3 -m tracks.legends` writes `out/legends/`:
- `<world>/tier1–3.ogg` and `tier1–3_tail.ogg`: loops and their ring-outs
- `<world>/<stinger>.ogg`
- `glims/glim_<midi>.ogg` and `king_<midi>.ogg`
- `kazoo_rumble.ogg`
- `legends.json`

`--web` also writes the player's sprites to `player/legends/`.

New instruments for the pack are in `src/orchestra/inst_legends.py`: `reed.kazoo`, `reed.kazoo_bass`, `voice.whistle`, `voice.teensy_babble`, `voice.teensy_cheer`, `pluck.jaw_harp`, `synth.slide_up`, `synth.slide_down`, `bell.glim` and `bell.glim_king`. The creature voices are wordless gibberish.
