/* Procedural space score. Shared stem IDs keep the adaptive conductor unchanged.
   Every area has a 16-bar progression on Canopy's grammar: bar 1 D-rooted, bars 8 and 16 A-rooted. */
'use strict';
window.CanopySpace = (() => {
  const areas = {
    A: ['Orbital Drift', 'weightless, warm, slowly unfolding',
      'Dmaj9 E/D Bm9 Gmaj7#11 Dmaj9/F# Em9 Gmaj9 Asus2 Gmaj7#11 F#m9 Em9 A7sus4 Bm9 Gmaj9 Em11 A9sus4'],
    B: ['Ion Nebula', 'glassy, luminous, flowing',
      'Dmaj7#11 Bm11 Gmaj9 E/D F#m11 Bm9 Gmaj7#11 Asus4 Dmaj9 E/D Bm11 Gmaj9 Em9 F#m7 Gmaj7#11 A9sus4'],
    C: ['Deep Field', 'wide, still, distant',
      'Dmaj9 Dmaj9 Bm11 Bm11 Gmaj9 Gmaj9 Em11 Asus2 Bm11 Bm11 Gmaj7#11 Gmaj7#11 Em9 Em9 F#m11 Asus4'],
    D: ['Derelict Station', 'hollow, mysterious, suspended',
      'Dm9 Bbmaj7#11 Gm9 Ebmaj7#11 Dm11 Fmaj7#11 Gm6/9 A7sus4 Bbmaj9 Gm11 Ebmaj7#11 Asus4 Dm9 Bbmaj7#11 Gm6/9 A7b9'],
    E: ['Warp Corridor', 'pulsing, bright, accelerating',
      'Dsus2 Cadd9 Bm7 Gadd9 Dsus2 Cadd9 Em7 Asus4 Bm7 Gadd9 Dsus2/F# Cadd9 Em9 Gadd9 F#m7 A7sus4'],
  };
  for (const a in areas) areas[a][2] = areas[a][2].split(' ');
  const labels = {amb:'Stellar drone', water:'Ion wash', wildlife:'Radio signals', birds:'Beacon theremin',
    woods:'Sonar', shaker:'Solar dust', drums:'Engine pulse', bass:'Sub bass', keys:'Glass keys',
    pad:'Nebula pad', marimba:'Sequencer', kalimba:'Starlight'};
  const rate = 24000, duration = 128 / 3, beat = 2 / 3, bar = 4 * beat;
  const hz = midi => 440 * 2 ** ((midi - 69) / 12);
  function random(seed){ return () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; }; }
  // Small chord parser for the labels above: root, slash bass and chord-tone pitch classes.
  const PC = {C:0, D:2, E:4, F:5, G:7, A:9, B:11};
  const QUAL = {'':[0,4,7], m7:[0,3,7,10], m9:[0,3,7,10,14], m11:[0,3,7,10,14,17], maj9:[0,4,7,11,14], 'maj7#11':[0,4,7,11,18],
    'm6/9':[0,3,7,9,14], sus2:[0,2,7], sus4:[0,5,7], '7sus4':[0,5,7,10], '9sus4':[0,5,7,10,14], add9:[0,4,7,14], '7b9':[0,4,7,10,13]};
  function chord(label){
    const [, r, q, b] = /^([A-G][#b]?)([^/]*(?:\/9)?)(?:\/([A-G][#b]?))?$/.exec(label);
    const pc = s => (PC[s[0]] + (s[1] === '#' ? 1 : s[1] === 'b' ? -1 : 0) + 12) % 12;
    const root = pc(r), iv = QUAL[q];
    if (!iv) throw new Error('Unknown chord ' + label);
    return {root, bass:b ? pc(b) : root, iv, pcs:[...new Set(iv.map(i => (root + i) % 12))]};
  }
  // Pads re-voice per chord with voice-leading: each tone takes the octave nearest the previous voicing.
  function voicings(area){
    let prev = [57, 62, 66, 69];
    return areas[area][2].map(label => {
      const c = chord(label), tones = c.iv.length > 4 ? c.iv.slice(1) : c.iv;
      const v = [...new Set(tones.map(i => {
        let best = 0, cost = 1e9;
        for (let m = 50; m <= 79; m++) if ((m - c.root - i) % 12 === 0) {
          const d = Math.min(...prev.map(p => Math.abs(p - m))) + Math.abs(m - 63) * .08;
          if (d < cost) { cost = d; best = m; }
        }
        return best;
      }))].sort((x, y) => x - y);
      prev = v; return v;
    });
  }
  function configure(meta){
    meta.leadin = {};
    for (const [id, [name, mood, chords]] of Object.entries(areas)) Object.assign(meta.sections[id], {name, mood, chords:[...chords]});
    return meta;
  }
  // Notes wrap into the start of the loop; separate tails let the conductor release them.
  function render(ctx, key){
    const [area, stem] = key.split('_'), prog = areas[area][2].map(chord), voiced = voicings(area);
    const n = Math.round(duration * rate), tailN = 9 * rate;
    const buffer = ctx.createBuffer(2, n, rate);
    const t16 = ctx.createBuffer(2, tailN, rate), t8 = ctx.createBuffer(2, tailN, rate);
    const channels = [buffer.getChannelData(0), buffer.getChannelData(1)];
    const tails = [[t16.getChannelData(0), t16.getChannelData(1)], [t8.getChannelData(0), t8.getChannelData(1)]];
    const rnd = random([...key].reduce((s, c) => s * 31 + c.charCodeAt(0) | 0, 17));
    const SLOW = {pad:[.9, 1.4], sub:[.05, .5], air:[1.5, 2], theremin:[.12, .5]};
    // freq may be a number or a pitch path [[t, hz], ...] (theremin glides between points)
    function note(at, length, freq, amp, pan, kind){
      const start = Math.round(at * rate), len = Math.round(length * rate);
      const gains = [Math.sqrt((1 - pan) / 2), Math.sqrt((1 + pan) / 2)];
      const [attack, release] = SLOW[kind] || [.004, .18];
      const path = Array.isArray(freq) ? freq : null;
      let phase = 0, mph = 0, low = 0, k = 0, f = path ? path[0][1] : freq;
      for (let j = 0; j < len; j++) {
        const t = j / rate;
        if (path) {
          while (k < path.length - 1 && t >= path[k + 1][0]) k++;
          const tgt = path[k][1]; f += (tgt - f) * .0009; // portamento
        }
        phase += 2 * Math.PI * f / rate;
        const env = Math.min(1, t / attack) * Math.min(1, (length - t) / release);
        let v;
        if (kind === 'pad') v = (Math.sin(phase) + .28 * Math.sin(phase * 2 + .3 * Math.sin(t * 1.7)) + .1 * Math.sin(phase * 3.003)) * .6;
        else if (kind === 'glass') { mph += 2 * Math.PI * f * 3.5 / rate; v = Math.sin(phase + 2.2 * Math.exp(-t * 4) * Math.sin(mph)) * Math.exp(-t * 1.3); }
        else if (kind === 'pulse') { const s = Math.sin(phase); v = Math.tanh(3 * s + .8 * Math.sin(phase * 2 + t * 5)) * .5 * Math.exp(-t * 7); }
        else if (kind === 'sub') v = (Math.sin(phase) + .15 * Math.sin(phase * 2)) * .9;
        else if (kind === 'engine') v = Math.sin(2 * Math.PI * 46 * t + 6 * (1 - Math.exp(-t * 26))) * Math.exp(-t * 7);
        else if (kind === 'sonar') v = Math.sin(phase * (1 - .02 * Math.min(1, t * 2))) * Math.exp(-t * 2.2);
        else if (kind === 'dust') v = (rnd() * 2 - 1) * Math.exp(-t * 26);
        else if (kind === 'air') { low += ((rnd() * 2 - 1) - low) * .02; v = low * 4 * (.6 + .4 * Math.sin(t * .7)); }
        else if (kind === 'theremin') v = (Math.sin(phase + .004 * f * Math.sin(t * 34)) + .12 * Math.sin(phase * 2)) * (.8 + .2 * Math.sin(t * 5.5));
        else v = Math.sin(phase + 1.5 * Math.sin(phase * 2) * Math.exp(-t * 3)) * Math.exp(-t * 1.5); // bell
        v *= amp * env;
        const pos = start + j;
        for (let c = 0; c < 2; c++) {
          const sample = v * gains[c];
          channels[c][pos % n] += sample;
          if (pos >= n && pos - n < tailN) tails[0][c][pos - n] += sample;
          if (start < n / 2 && pos >= n / 2 && pos - n / 2 < tailN) tails[1][c][pos - n / 2] += sample;
        }
      }
    }
    const tone = (c, i, base) => { const pcs = c.iv.map(x => (c.root + x) % 12); let m = base; while (m % 12 !== pcs[i % pcs.length]) m++; return m; };
    const take = stem.startsWith('amb') ? +stem.slice(3) || 1 : 0;
    const drive = area === 'E', dark = area === 'D';
    if (stem.startsWith('amb')) {
      // D-A pedal drone with a slow noise breath; takes differ in pan and colour.
      for (let b = 0; b < 64; b += 8) {
        [38, 45].forEach((p, i) => note(b * beat, 8 * beat + 2, hz(p), .028, (i ? 1 : -1) * (.2 + .15 * take), 'pad'));
        note(b * beat, 8 * beat + 2, hz(dark ? 53 : 54 + 12 * (take % 2)), .006, rnd() - .5, 'pad');
      }
      note(0, duration + 2, 0, .004 + .001 * take, 0, 'air');
    } else if (stem === 'water') {
      // ion wash: high chord colours per bar over a slow noise swell
      prog.forEach((c, bar_) => [1, 3].forEach((i, j) => note(bar_ * bar, bar + 1.6, hz(tone(c, i, 74) + 12 * j), .007, j ? .6 : -.6, 'pad')));
      for (let b = 0; b < 64; b += 8) note(b * beat, 8 * beat + 1, 0, .006, rnd() - .5, 'air');
    } else if (stem === 'pad') {
      voiced.forEach((v, b) => {
        if (b > 0 && voiced[b - 1].join() === v.join()) return; // a held chord sustains
        let len = 1; while (b + len < 16 && voiced[b + len].join() === v.join()) len++;
        v.forEach((p, i) => note(b * bar, len * bar + 1.4, hz(p), .013, (i % 2 ? 1 : -1) * (.25 + .1 * i), 'pad'));
      });
    } else if (stem === 'bass') {
      prog.forEach((c, b) => {
        const root = 36 + ((c.bass - 0 + 12) % 12), r = root > 45 ? root - 12 : root;
        if (drive) for (let e = 0; e < 8; e++) note(b * bar + e * beat / 2, beat / 2 - .02, hz(r + (e % 4 === 3 ? 12 : 0)), .05, 0, 'pulse');
        else { note(b * bar, 3 * beat, hz(r), .05, 0, 'sub'); note(b * bar + 3 * beat, beat, hz(r + (dark ? 7 : 12)), .03, 0, 'sub'); }
      });
    } else if (stem === 'keys') {
      prog.forEach((c, b) => [0, 2.5].forEach((o, j) => voiced[b].slice(j).forEach((p, i) =>
        note(b * bar + o * beat + i * .03, 2.4, hz(p + 12), .011, (i % 2 ? .5 : -.5), 'glass'))));
    } else if (stem === 'marimba') {
      // sequencer arpeggiates the bar's chord
      const step = drive ? .25 : .5, shape = [0, 1, 2, 3, 2, 1, 3, 4];
      prog.forEach((c, b) => { for (let s = 0; s < 4 / step; s++) note(b * bar + s * step * beat, .35, hz(tone(c, shape[s % 8], 62)), drive ? .02 : .024, s % 2 ? .35 : -.35, 'pulse'); });
    } else if (stem === 'birds') {
      // beacon theremin: fragments of Canopy's A-D-E-A motif (5th, 8ve, 9th, 5th up), transposed to each chord root
      const motif = [7, 12, 14, 19], frags = [[0, 1], [1, 2, 3], [0, 1, 2], [3, 2, 1]];
      for (let b = 1; b < 16; b += 2) {
        const c = prog[b], fr = frags[(b >> 1) % 4], base = 60 + c.root - (c.root > 5 ? 12 : 0);
        const path = fr.map((m, i) => [i * beat * .75, hz(base + motif[m])]);
        note(b * bar + beat, fr.length * beat * .75 + .9, path, .022, (b % 4 === 1 ? -.4 : .4), 'theremin');
        note(b * bar + beat, 2.5, hz(base + motif[fr[0]] + 12), .01, .2, 'glass');
      }
    } else if (stem === 'kalimba') {
      for (let b = 0; b < 64; b += 2) note((b + .5) * beat, 1.6, hz(tone(prog[b >> 2], Math.floor(rnd() * 5), 84)), .012, rnd() * 1.4 - .7, 'glass');
    } else if (stem === 'wildlife') {
      // radio signals: short gliding blips tuned to the chord
      for (let b = 0; b < 62; b += 6) {
        const c = prog[b >> 2], p = tone(c, Math.floor(rnd() * 4), 79);
        note((b + rnd()) * beat, .5, [[0, hz(p)], [.15, hz(p + 5)], [.3, hz(p)]], .012, rnd() * 1.4 - .7, 'theremin');
      }
    } else if (stem === 'woods') {
      prog.forEach((c, b) => note(b * bar + (b % 2 ? 2 * beat : 0), 2.2, hz(48 + c.root), .03, rnd() - .5, 'sonar'));
    } else if (stem === 'shaker') {
      for (let s = 0; s < 64 * (drive ? 4 : 2); s++) note(s * beat / (drive ? 4 : 2), .16, 0, s % 2 ? .009 : .016, rnd() - .5, 'dust');
    } else if (stem === 'drums') {
      for (let b = 0; b < 64; b++) if (drive || b % 2 === 0) note(b * beat, .8, 0, b % 4 === 0 ? .075 : .055, 0, 'engine');
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
  return {areas, labels, chord, voicings, configure, render, frame};
})();
