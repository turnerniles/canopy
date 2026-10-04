/* Procedural space score. Shared stem IDs keep the adaptive conductor unchanged. */
'use strict';
window.CanopySpace = (() => {
  const areas = {
    A: ['Orbital Drift', 'weightless, warm, slowly unfolding', [50, 57, 61, 64, 69], 'Dmaj9'],
    B: ['Ion Nebula', 'glassy, luminous, flowing', [55, 62, 66, 69, 73], 'Gmaj9#11'],
    C: ['Deep Field', 'wide, still, distant', [47, 54, 57, 61, 64], 'Bm11'],
    D: ['Derelict Station', 'hollow, mysterious, suspended', [50, 57, 60, 64, 69], 'Dm9'],
    E: ['Warp Corridor', 'pulsing, bright, accelerating', [45, 52, 57, 59, 64], 'Asus2'],
  };
  const labels = {amb:'Stellar drone', water:'Ion wash', wildlife:'Radio signals', birds:'Beacon chimes',
    woods:'Sonar', shaker:'Solar dust', drums:'Engine pulse', bass:'Sub bass', keys:'Glass keys',
    pad:'Nebula pad', marimba:'Sequencer', kalimba:'Starlight'};
  const rate = 24000, duration = 128 / 3, beat = 2 / 3;
  const hz = midi => 440 * 2 ** ((midi - 69) / 12);
  function random(seed){ return () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; }; }
  function configure(meta){
    meta.leadin = {};
    for (const [id, [name, mood, , chord]] of Object.entries(areas)) {
      Object.assign(meta.sections[id], {name, mood, chords:Array(16).fill(chord)});
    }
    return meta;
  }
  // Notes wrap into the start of the loop; separate tails let the conductor release them.
  function render(ctx, key){
    const [area, stem] = key.split('_'), notes = areas[area][2];
    const n = Math.round(duration * rate), tailN = 9 * rate;
    const buffer = ctx.createBuffer(2, n, rate);
    const t16 = ctx.createBuffer(2, tailN, rate), t8 = ctx.createBuffer(2, tailN, rate);
    const channels = [buffer.getChannelData(0), buffer.getChannelData(1)];
    const rnd = random([...key].reduce((s, c) => s * 31 + c.charCodeAt(0) | 0, 17));
    function note(at, length, freq, amp, pan, kind){
      const start = Math.round(at * rate), len = Math.round(length * rate);
      const gains = [Math.sqrt((1 - pan) / 2), Math.sqrt((1 + pan) / 2)];
      const attack = kind === 'pad' ? .9 : .025;
      for (let j = 0; j < len; j++) {
        const t = j / rate, phase = 2 * Math.PI * freq * t;
        const env = Math.min(1, t / attack) * Math.min(1, (length - t) / (kind === 'pad' ? 1.4 : .18));
        let v;
        if (kind === 'pad') v = (Math.sin(phase) + .28 * Math.sin(phase * 2 + .3 * Math.sin(t * 1.7))) * .65;
        else if (kind === 'dust') v = (rnd() * 2 - 1) * Math.exp(-t * 22);
        else if (kind === 'pulse') v = Math.sin(phase + 5 * (1 - Math.exp(-t * 24))) * Math.exp(-t * 6);
        else v = Math.sin(phase + 1.5 * Math.sin(phase * 2) * Math.exp(-t * 3)) * Math.exp(-t * 1.5);
        v *= amp * env;
        const pos = start + j;
        for (let c = 0; c < 2; c++) {
          const sample = v * gains[c];
          channels[c][pos % n] += sample;
          if (pos >= n && pos - n < tailN) t16.getChannelData(c)[pos - n] += sample;
          if (start < n / 2 && pos >= n / 2 && pos - n / 2 < tailN) t8.getChannelData(c)[pos - n / 2] += sample;
        }
      }
    }
    if (stem.startsWith('amb') || stem === 'water' || stem === 'pad') {
      const wash = stem === 'water', ambient = stem.startsWith('amb');
      for (let b = 0; b < 64; b += 8) {
        const chord = ambient ? [notes[0] - 12, notes[1] - 12] : wash ? [notes[2] + 12, notes[4] + 12] : notes;
        chord.forEach((pitch, i) => note(b * beat, 8 * beat + 2, hz(pitch), ambient ? .025 : wash ? .008 : .013, (i % 2 ? 1 : -1) * (.3 + rnd() * .5), 'pad'));
      }
    } else {
      const step = {wildlife:8, birds:8, woods:4, shaker:.5, drums:2, bass:8, keys:4, marimba:1, kalimba:2}[stem];
      for (let b = 0; b < 64; b += step) {
        const pitch = stem === 'bass' ? notes[0] - 12 : notes[Math.floor(rnd() * notes.length)] + (['birds','kalimba','wildlife'].includes(stem) ? 12 : 0);
        const kind = stem === 'shaker' ? 'dust' : stem === 'drums' ? 'pulse' : stem === 'bass' ? 'pad' : 'bell';
        const length = stem === 'bass' ? step * beat + 1 : kind === 'dust' ? .18 : kind === 'pulse' ? .8 : 3;
        note(b * beat, length, stem === 'drums' ? 48 : hz(pitch), stem === 'bass' ? .045 : kind === 'pulse' ? .065 : .025, stem === 'bass' || stem === 'drums' ? 0 : rnd() * 1.5 - .75, kind);
      }
    }
    return {buffer, tails:{t16, t8}};
  }
  function frame(canvas, glow, time, area, intensity){
    const g = canvas.getContext('2d'), gg = glow.getContext('2d');
    gg.clearRect(0, 0, 320, 140);
    const index = Object.keys(areas).indexOf(area);
    g.fillStyle = ['#080e25','#170f30','#070d1c','#101020','#0a122c'][index]; g.fillRect(0,0,320,140);
    const rnd = random(9876);
    for(let i=0;i<150;i++) {
      const speed = .1 + rnd() * (.6 + intensity * 2);
      const x = ((rnd() * 320 - time * speed) % 320 + 320) % 320, y = rnd() * 140;
      g.fillStyle = `rgba(200,220,255,${.3 + .6 * (.5 + .5 * Math.sin(time * .4 + i))})`;
      g.fillRect(Math.floor(x),Math.floor(y),i % 13 === 0 ? 2 : 1,1);
    }
    g.fillStyle = ['#2e4274','#75518c','#384364','#454359','#305881'][index];
    g.beginPath(); g.arc(238, 67, 33, 0, Math.PI * 2); g.fill();
    g.fillStyle = '#111a32'; g.beginPath(); g.arc(249,60,31,0,Math.PI*2); g.fill();
    g.strokeStyle = '#93b8d0'; g.lineWidth = 1;
    g.beginPath(); g.ellipse(238,67,49,9,-.3,0,Math.PI*2); g.stroke();
    const y = 83 + Math.round(Math.sin(time * .3) * 3);
    g.fillStyle='#afcce1'; g.fillRect(73,y,16,3); g.fillRect(77,y-3,7,9);
    g.fillStyle='#72d8ff'; g.fillRect(69,y,4,3);
  }
  return {areas, labels, configure, render, frame};
})();
