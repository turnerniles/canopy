const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, 'space.js'), 'utf8');
const synth = new Function('window', source + '\nreturn window.CanopySpace;')({});
const ctx = {createBuffer(channels, length, sampleRate) {
  const data = Array.from({length:channels}, () => new Float32Array(length));
  return {length, sampleRate, getChannelData:c => data[c]};
}};
const RATE = 24000, BAR = RATE * 8 / 3;
// Shared grammar: bar 1 D-rooted, bars 8 and 16 A-rooted; real progressions with per-bar movement.
const meta = synth.configure(JSON.parse(fs.readFileSync(path.join(__dirname, 'meta.json'), 'utf8')));
for (const area of Object.keys(synth.areas)) {
  const labels = meta.sections[area].chords, chords = labels.map(synth.chord);
  assert.equal(labels.length, 16);
  assert.equal(chords[0].root, 2, `${area} bar 1 must be D-rooted`);
  assert.equal(chords[7].root, 9, `${area} bar 8 must be A-rooted`);
  assert.equal(chords[15].root, 9, `${area} bar 16 must be A-rooted`);
  assert.ok(new Set(labels).size >= 6, `${area} needs a real progression`);
  const v = synth.voicings(area);
  for (let b = 1; b < 16; b++) {
    if (labels[b] !== labels[b - 1]) assert.notDeepEqual(v[b], v[b - 1], `${area} bar ${b + 1} re-voices`);
    v[b].forEach(m => assert.ok(chords[b].pcs.includes(m % 12), `${area} bar ${b + 1} pad note in chord`));
    const moved = v[b].reduce((s, m) => s + Math.min(...v[b - 1].map(p => Math.abs(p - m))), 0) / v[b].length;
    assert.ok(moved <= 2.5, `${area} bar ${b + 1} voice-leading moves ${moved} semitones on average`);
  }
}
const minor = synth.chord(meta.sections.D.chords[0]).pcs.includes(5), major = synth.chord(meta.sections.A.chords[0]).pcs.includes(6);
assert.ok(minor && major, 'Derelict Station is D minor; Orbital Drift is D major');
// Goertzel power of one pitch over a window
function power(data, start, len, freq) {
  const w = 2 * Math.cos(2 * Math.PI * freq / RATE); let s1 = 0, s2 = 0;
  for (let i = 0; i < len; i++) { const s0 = data[start + i] + w * s1 - s2; s2 = s1; s1 = s0; }
  return s1 * s1 + s2 * s2 - w * s1 * s2;
}
const pcPower = (data, start, len, lo, hi) => Array.from({length:12}, (_, pc) => {
  let p = 0; for (let m = lo; m <= hi; m++) if (m % 12 === pc) p += power(data, start, len, 440 * 2 ** ((m - 69) / 12)); return p;
});
let peak = 0, maxSeam = 0, checkedBars = 0;
for (const area of Object.keys(synth.areas)) {
  const mix = new Float32Array(1024000);
  for (const stem of [...Object.keys(synth.labels), 'amb2', 'amb3', 'amb4']) {
    const {buffer, tails} = synth.render(ctx, `${area}_${stem}`);
    assert.equal(buffer.length, 1024000);
    for (let c=0;c<2;c++) {
      const data = buffer.getChannelData(c);
      let energy=0;
      for(let i=0;i<data.length;i++) {
        const value = data[i];
        assert.ok(Number.isFinite(value));
        peak = Math.max(peak, Math.abs(value)); energy += value * value;
        if (c === 0 && !['amb2','amb3','amb4'].includes(stem)) mix[i] += value;
      }
      assert.ok(energy > .01, `${area}_${stem} must be audible`);
      const seam = Math.abs(data[0] - data[data.length-1]);
      maxSeam = Math.max(maxSeam,seam);
      assert.ok(seam < .025, `${area}_${stem} loop seam ${seam}`);
      assert.equal(tails.t16.length, 216000);
      assert.equal(tails.t8.length, 216000);
      // Only the previous pass's wrapped release sounds at sample zero.
      assert.ok(Math.abs(data[0] - tails.t16.getChannelData(c)[0]) < 1e-6);
      assert.ok(Math.abs(data[512000] - tails.t8.getChannelData(c)[0]) < 1e-5, `${area}_${stem} half-loop release matches tail`);
    }
    if (stem === 'pad' || stem === 'bass') {
      // The pad's strongest pitch classes each bar are its chord; the bass sits on the bar's bass note.
      const data = buffer.getChannelData(0), chords = meta.sections[area].chords.map(synth.chord);
      chords.forEach((ch, b) => {
        const p = stem === 'pad' ? pcPower(data, Math.round(b * BAR + 1.45 * RATE), RATE / 2, 50, 79) : pcPower(data, Math.round(b * BAR + .2 * RATE), RATE / 2, 33, 47);
        const top = p.map((x, i) => [x, i]).sort((x, y) => y[0] - x[0]);
        if (stem === 'pad') top.slice(0, 3).forEach(([, pc]) => assert.ok(ch.pcs.includes(pc), `${area} pad bar ${b + 1}: pc ${pc} not in ${meta.sections[area].chords[b]}`));
        else assert.equal(top[0][1], ch.bass, `${area} bass bar ${b + 1} follows the root`);
        checkedBars++;
      });
    }
  }
  for (const v of mix) assert.ok(Math.abs(v) < .95, `${area} full-mix headroom`);
}
assert.ok(peak < .2, `Stem headroom: ${peak}`);
const first = synth.render(ctx,'A_birds').buffer.getChannelData(0);
const second = synth.render(ctx,'A_birds').buffer.getChannelData(0);
assert.deepEqual(first,second);
console.log(`75 stems passed: finite, audible, deterministic, seamless, mix headroom; grammar, voice-leading and ${checkedBars} pad/bass bars follow the progression; peak ${peak.toFixed(4)}, max seam ${maxSeam.toFixed(4)}`);
