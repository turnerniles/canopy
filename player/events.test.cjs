const assert = require('node:assert/strict');
require('./events.js');
const E = CanopyEvents;
const near = (a, b, msg) => assert.ok(Math.abs(a - b) < 1e-9, `${msg}: ${a} vs ${b}`);
const onGrid = (t, origin, step, msg) => near(Math.round((t - origin) / step) * step + origin, t, msg);
// chord parser covers every label the soundtracks use
const fs = require('node:fs'), path = require('node:path');
const meta = JSON.parse(fs.readFileSync(path.join(__dirname, 'meta.json'), 'utf8'));
global.window = {}; require('./space.js'); require('./fusion.js');
const labels = new Set();
for (const a of 'ABCDE') meta.sections[a].chords.forEach(c => labels.add(c));
for (const a in window.CanopySpace.areas) window.CanopySpace.areas[a][2].forEach(c => labels.add(c));
for (const m of Object.values(CanopyFusion.modes)) m.harmony.forEach(c => labels.add(c));
for (const l of labels) {
  const c = E.parse(l);
  assert.ok(c.pcs.length >= 3 && c.pcs.every(p => p >= 0 && p < 12), `parse ${l}`);
}
assert.deepEqual(E.parse('Dmaj9').pcs.sort((a, b) => a - b), [1, 2, 4, 6, 9]);
assert.deepEqual(E.parse('Gm6/9').pcs.sort((a, b) => a - b), [2, 4, 7, 9, 10]);
assert.equal(E.parse('E/D').bass, 2);
assert.equal(E.parse('Asus2→A', .75).tones.join(), '9,1,4', 'second half of a split bar');
assert.deepEqual(E.parse('A7sus4♭9').pcs.sort((a, b) => a - b), [2, 4, 7, 9, 10]);
// space/fusion chord grammar agrees with events.js
for (const a in window.CanopySpace.areas) for (const l of window.CanopySpace.areas[a][2])
  assert.deepEqual([...E.parse(l).tones].sort(), window.CanopySpace.chord(l).pcs.sort(), `space ${l}`);

const bpm = 90, beat = 60 / bpm, bar = 4 * beat, s16 = beat / 4, origin = 10.1;
const base = {bpm, origin, chord:'Bm11', key:'D major', palette:{lead:'kalimba'}};
const inChord = (notes, label, msg) => notes.forEach(n => assert.ok(E.parse(label).pcs.includes(n.midi % 12), `${msg}: ${n.midi} in ${label}`));
for (const now of [10.1, 10.37, 12.0001, 17.9]) {
  for (const name of E.NAMES) {
    const s = E.schedule(name, {...base, now});
    assert.ok(s.at >= now, `${name} is in the future`);
    assert.ok(s.at - now <= (name === 'victory' ? bar + .05 : name === 'checkpoint' ? beat + 3 * s16 + .02 : beat + .02), `${name} lands promptly`);
    onGrid(s.at, origin, name === 'victory' ? bar : ['checkpoint','respawn','boss','boss_end'].includes(name) ? beat : s16, `${name} quantized`);
    s.notes.forEach(n => { assert.ok(n.time >= now - 1e-9 && n.dur > 0 && n.gain > 0 && n.gain < .5 && Number.isInteger(n.midi) && n.voice, `${name} note shape`); });
    if (['collect','checkpoint','secret'].includes(name)) inChord(s.notes, 'Bm11', name);
  }
}
assert.equal(E.schedule('jump', {...base, now:11}).notes.length, 0, 'jump is silent by default');
assert.equal(E.schedule('jump', {...base, now:11, opts:{sound:true}}).notes.length, 1);
// collect: ascends within the chain, follows chord changes, resets after 1.2 s idle
let chain = null, last = -1, t = 11;
for (let i = 0; i < 6; i++, t += .3) {
  const chord = i < 3 ? 'Dmaj9' : 'Gmaj9';
  const s = E.schedule('collect', {...base, chord, now:t, chain});
  const m = s.notes[0].midi; inChord([s.notes[0]], chord, 'collect chain');
  assert.ok(m > last, `collect ${i} ascends (${m} > ${last})`); last = m; chain = s.chain;
  onGrid(s.at, origin, s16, 'collect on the 16th');
  assert.ok(s.at - t < s16 + .02, 'collect is nearly immediate');
}
assert.equal(chain.count, 6);
const first = E.schedule('collect', {...base, chord:'Dmaj9', now:11}).notes[0].midi;
const after = E.schedule('collect', {...base, chord:'Dmaj9', now:t + 1.3, chain});
assert.equal(after.notes[0].midi, first, 'chain resets after idle'); assert.equal(after.chain.count, 1);
const capped = Array.from({length:30}).reduce(c => E.schedule('collect', {...base, now:11, chain:c}).chain, null);
assert.ok(capped.last <= 98, 'chain holds at the top of the register');
// checkpoint: rising arpeggio whose top note lands on the beat
const cp = E.schedule('checkpoint', {...base, now:11}), arp = cp.notes.filter(n => n.time < cp.at + 1e-9 && n.midi > 50).slice(0, 4);
for (let i = 1; i < arp.length; i++) assert.ok(arp[i].midi > arp[i - 1].midi && arp[i].time > arp[i - 1].time, 'checkpoint rises');
near(arp[arp.length - 1].time, cp.at, 'arpeggio lands on the beat');
// victory: on the next bar line, 2 bars long, in key, ducks the bed
for (const [key, scale] of [['D major', [2,4,6,7,9,11,1]], ['D minor', [2,4,5,7,9,10,0]], ['G major', [7,9,11,0,2,4,6]]]) {
  const v = E.schedule('victory', {...base, key, now:12.3});
  onGrid(v.at, origin, bar, 'victory on a bar line'); assert.ok(v.at > 12.3 && v.at - 12.3 <= bar + .05);
  v.notes.forEach(n => assert.ok(scale.includes(n.midi % 12), `victory ${key} note ${n.midi} in key`));
  const end = Math.max(...v.notes.map(n => n.time + n.dur));
  assert.ok(end - v.at > 1.6 * bar && end - v.at < 2.4 * bar, 'two-bar fanfare');
  assert.ok(v.fx.some(f => f.type === 'duck' && f.bus === 'music' && f.hold > bar), 'bed ducks under the fanfare');
}
// death: a falling figure plus a one-bar fade of the music bus; respawn restores it
const d = E.schedule('death', {...base, now:13}), fall = d.notes.filter(n => n.voice === 'kalimba');
assert.ok(fall.length >= 4);
for (let i = 1; i < fall.length; i++) assert.ok(fall[i].midi < fall[i - 1].midi && fall[i].time > fall[i - 1].time, 'death falls');
assert.ok(fall.some(n => (n.midi - E.parse('Bm11').root) % 12 === 3), 'minor third in the fall');
const fade = d.fx.find(f => f.type === 'fade'); near(fade.dur, bar, 'fade over one bar'); assert.ok(fade.cutoff < 500);
assert.ok(E.schedule('respawn', {...base, now:20}).fx.some(f => f.type === 'restore'));
// hurt ducks for about 0.6 s with a low-pass; boss toggles
const h = E.schedule('hurt', {...base, now:13}).fx[0];
assert.ok(h.type === 'duck' && h.bus === 'all' && Math.abs(h.hold + h.release - .6) < .05 && h.cutoff < 1000);
assert.equal(E.schedule('boss', {...base, now:13}).fx[0].on, true);
assert.equal(E.schedule('boss_end', {...base, now:13}).fx[0].on, false);
// secret: an ascending glissando
const sec = E.schedule('secret', {...base, now:13}).notes;
for (let i = 1; i < sec.length; i++) assert.ok(sec[i].midi > sec[i - 1].midi);
// synthesized area transition lands on the boundary
const tr = E.schedule('transition', {...base, chord:'Dmaj9', now:14, time:origin + 8 * bar, lead:bar, variant:'shimmer'});
near(Math.max(...tr.notes.filter(n => n.voice === 'swell').map(n => n.time + n.dur)), origin + 8 * bar, 'riser ends on the downbeat');
assert.ok(tr.notes.some(n => n.time >= origin + 8 * bar - 1e-9 && n.voice === 'kalimba'), 'landing chime on the downbeat');
assert.throws(() => E.schedule('explode', base), TypeError);
// palettes and the Web Audio renderer (against a recording mock)
assert.equal(E.palette('jungle').lead, 'kalimba'); assert.equal(E.palette('space').lead, 'glass'); assert.equal(E.palette('canopy').lead, 'steel');
const calls = [];
const param = () => ({value:0, setValueAtTime(v, t){ assert.ok(Number.isFinite(v) && Number.isFinite(t)); calls.push(t); }, linearRampToValueAtTime(v, t){ assert.ok(Number.isFinite(v)); },
  exponentialRampToValueAtTime(v, t){ assert.ok(v > 0 && Number.isFinite(t)); }, setTargetAtTime(){}, cancelScheduledValues(){}});
const node = extra => ({connect(){}, disconnect(){}, gain:param(), frequency:param(), Q:param(), start(t){ assert.ok(Number.isFinite(t)); }, stop(t){ assert.ok(Number.isFinite(t)); }, ...extra});
const mock = {currentTime:14, sampleRate:48000, createGain:() => node(), createOscillator:() => node({type:'sine'}), createBiquadFilter:() => node({type:'lowpass'}),
  createBufferSource:() => node({buffer:null, loop:false}), createBuffer:(c, n) => ({getChannelData:() => new Float32Array(n)})};
let played = 0;
for (const pal of ['kalimba','glass','steel','mallet']) for (const name of [...E.NAMES, 'transition'])
  played += E.play(mock, node(), E.schedule(name, {...base, palette:{lead:pal}, now:14, time:16, lead:1, opts:{sound:true}}).notes);
assert.ok(played > 200 && calls.length > played);
console.log(`events passed: ${labels.size} chord labels parsed; quantization, in-chord notes, collect chain, checkpoint, victory, death fall, hurt duck, transitions; ${played} voices rendered against a Web Audio mock`);
