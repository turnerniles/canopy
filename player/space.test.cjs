const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, 'space.js'), 'utf8');
const synth = new Function('window', source + '\nreturn window.CanopySpace;')({});
const ctx = {createBuffer(channels, length, sampleRate) {
  const data = Array.from({length:channels}, () => new Float32Array(length));
  return {length, sampleRate, getChannelData:c => data[c]};
}};
let peak = 0, maxSeam = 0;
for (const area of Object.keys(synth.areas)) {
  for (const stem of [...Object.keys(synth.labels), 'amb2', 'amb3', 'amb4']) {
    const {buffer, tails} = synth.render(ctx, `${area}_${stem}`);
    assert.equal(buffer.length, 1024000);
    for (let c=0;c<2;c++) {
      const data = buffer.getChannelData(c);
      let energy=0;
      for(const value of data) {
        assert.ok(Number.isFinite(value));
        peak = Math.max(peak, Math.abs(value)); energy += value * value;
      }
      assert.ok(energy > .01, `${area}_${stem} must be audible`);
      const seam = Math.abs(data[0] - data[data.length-1]);
      maxSeam = Math.max(maxSeam,seam);
      assert.ok(seam < .025, `${area}_${stem} loop seam ${seam}`);
      assert.equal(tails.t16.length, 216000);
      assert.equal(tails.t8.length, 216000);
      // Only the previous pass's wrapped release sounds at sample zero.
      assert.ok(Math.abs(data[0] - tails.t16.getChannelData(c)[0]) < 1e-6);
    }
  }
}
assert.ok(peak < .2, `Stem headroom: ${peak}`);
const first = synth.render(ctx,'A_birds').buffer.getChannelData(0);
const second = synth.render(ctx,'A_birds').buffer.getChannelData(0);
assert.deepEqual(first,second);
console.log(`75 stems passed: finite, audible, deterministic, seamless; peak ${peak.toFixed(4)}, max seam ${maxSeam.toFixed(4)}`);
