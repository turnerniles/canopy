/* Beat-quantized, in-key game events. schedule() is pure (Node-testable); play() renders the notes with Web Audio. */
'use strict';
globalThis.CanopyEvents = (() => {
  const NAMES = ['collect','jump','checkpoint','hurt','death','respawn','victory','secret','boss','boss_end'];
  const NOTE = {C:0, D:2, E:4, F:5, G:7, A:9, B:11};
  const acc = s => /[#♯]/.test(s || '') ? 1 : /[b♭]/.test(s || '') ? -1 : 0;
  const pcOf = s => (NOTE[s[0]] + acc(s[1]) + 12) % 12;
  // Chord-name parser for the soundtrack labels: Dmaj9, E/D, Bm11, A13sus, Gm6/9, A7sus4♭9, Asus4/9, Asus2→A ...
  function parse(label, frac = 0){
    if (label && typeof label === 'object') return Array.isArray(label) ? {root:label[0] % 12, bass:label[0] % 12, pcs:[...new Set(label.map(p => p % 12))], tones:label.map(p => p % 12)} : label;
    const parts = String(label).split(/→|->/).map(s => s.trim()).filter(Boolean);
    const m = /^([A-G][#b♯♭]?)(.*)$/.exec(parts[Math.min(parts.length - 1, Math.floor(Math.max(0, frac) * parts.length))] || '');
    if (!m) throw new TypeError('Unrecognised chord ' + label);
    const root = pcOf(m[1]); let q = m[2], bass = root;
    const sl = /\/([A-G][#b♯♭]?)$/.exec(q); if (sl){ bass = pcOf(sl[1]); q = q.slice(0, sl.index); }
    const iv = [0];
    if (/^5$/.test(q)) iv.push(7);
    else {
      const maj = /maj|Δ/.test(q), minor = /^m(?!aj)/.test(q), sus2 = /sus2/.test(q), sus4 = /sus(?!2)/.test(q);
      iv.push(sus2 ? 2 : sus4 ? 5 : minor ? 3 : 4, /dim|°|[b♭]5/.test(q) ? 6 : /aug|\+|[#♯]5/.test(q) ? 8 : 7);
      const n = Math.max(0, ...(q.replace(/sus\d?|add\d+|[#♯b♭]\d+|6\/9|\/\d+/g, '').match(/\d+/g) || []).map(Number));
      if (/6/.test(q.replace(/1[13]/g, '')) && n < 7) iv.push(9);
      if (n >= 7) iv.push(maj ? 11 : 10);
      if (n >= 9 || /6\/9|add9|\/9/.test(q)) iv.push(2);
      if (n >= 11 && !sus4) iv.push(5);
      if (n >= 13) iv.push(9);
      if (/[#♯]11/.test(q)) iv.push(6);
      if (/[b♭]9/.test(q)) iv.push(1);
      if (/[#♯]9/.test(q)) iv.push(3);
    }
    const tones = [...new Set(iv.map(i => (root + i) % 12))];
    return {root, bass, tones, pcs:[...new Set([...tones, bass])]};
  }
  function key(name){
    const k = String(name || 'D major').trim(), m = /^[A-G][#b♯♭]?/.exec(k);
    return {tonic:m ? pcOf(m[0]) : 2, minor:/minor|^[A-G][#b♯♭]?m(?!aj)/i.test(k)};
  }
  function palette(mode){ return {lead:mode === 'jungle' ? 'kalimba' : mode === 'space' ? 'glass' : ['mine','underwater','crystal'].includes(mode) ? 'mallet' : 'steel'}; }
  // chord pitches in [lo, hi], optionally thinned so neighbours are at least `gap` semitones apart
  function ladder(pcs, lo, hi, gap = 1){
    const out = [];
    for (let m = lo; m <= hi; m++) if (pcs.includes(m % 12) && (!out.length || m - out[out.length - 1] >= gap)) out.push(m);
    return out;
  }
  // Pure scheduler: grid {bpm, now, origin, beatsPerBar}; chord label/pcs (or t => label); returns {at, notes, fx, chain}.
  function schedule(name, a = {}){
    const beat = 60 / (a.bpm || 90), bar = beat * (a.beatsPerBar || 4), s16 = beat / 4;
    const now = a.now || 0, origin = a.origin || 0, lead = (a.palette || {}).lead || 'kalimba', opts = a.opts || {};
    const q = (t, step) => origin + Math.ceil((t - origin) / step - 1e-6) * step;
    const at = name === 'victory' ? q(now + .03, bar) : name === 'checkpoint' ? q(now + 3 * s16 + .015, beat)
      : ['respawn','boss','boss_end'].includes(name) ? q(now + .015, beat) : name === 'transition' ? a.time : q(now + .012, s16);
    const ch = parse(typeof a.chord === 'function' ? a.chord(at) : (a.chord || 'Dmaj9'));
    const r = ch.root, pcs = ch.pcs, k = key(a.key), notes = [], fx = [];
    const n = (time, midi, dur, gain, voice = lead) => notes.push({time, midi, dur, gain, voice});
    let chain = a.chain || null;
    if (name === 'collect'){
      const list = ladder(pcs, 74, 98, 2);
      const reset = !chain || chain.last == null || now - chain.at > 1.2;
      const midi = reset ? list[0] : (list.find(m => m > chain.last) ?? list[list.length - 1]);
      n(at, midi, .9, .22); n(at, midi + 12, .35, .05);
      chain = {last:midi, at:now, count:reset ? 1 : chain.count + 1};
    } else if (name === 'jump'){
      if (opts.sound) n(at, 96 + r > 104 ? 84 + r : 96 + r, .05, .045, 'tick');
    } else if (name === 'checkpoint'){
      const arp = ladder(pcs, 60 + r, 84 + r, 2).slice(0, 4);
      arp.forEach((m, i) => n(at - (arp.length - 1 - i) * s16, m, i === arp.length - 1 ? 1.4 : .5, i === arp.length - 1 ? .2 : .12));
      n(at, arp[arp.length - 1] + 12, .9, .07); n(at, 48 + r, 1.2, .1);
    } else if (name === 'hurt'){
      n(at, 36 + r, .35, .32, 'thud'); n(at, 61 + r, .22, .045, 'buzz'); n(at, 66 + r, .22, .04, 'buzz');
      fx.push({type:'duck', bus:'all', time:at, hold:.25, release:.35, gain:.5, cutoff:600});
    } else if (name === 'death'){
      [12, 10, 7, 3, 2, 0].forEach((iv, i) => n(at + i * beat / 2, 60 + r + iv, i === 5 ? 2.2 : .6, .17 - .015 * i));
      n(at + 5 * beat / 2, 36 + r, .8, .25, 'thud');
      fx.push({type:'fade', bus:'music', time:at, dur:bar, cutoff:250});
    } else if (name === 'respawn'){
      [0, 7, 12].forEach((iv, i) => n(at + i * s16, 60 + r + iv, .8, .1));
      fx.push({type:'restore', bus:'music', time:at, dur:1.5});
    } else if (name === 'victory'){
      const b = 60 + k.tonic, mi = k.minor;
      const tune = [[0,0,.5],[.5,mi?3:4,.5],[1,7,.5],[1.5,12,2.2],[4,mi?8:9,.5],[4.5,mi?10:11,.5],[5,14,1],[6,12,2.2]];
      tune.forEach(([t, iv, d]) => n(at + t * beat, b + iv, d * beat, .15, 'brass'));
      n(at + 1.5 * beat, b + 24, 1.5, .07); n(at + 6 * beat, b + 24, 2, .08);
      [[0, [b - 12, b - 5]], [6, [b - 12, b, b + (mi ? 3 : 4), b + 7]]].forEach(([t, ms]) => ms.forEach(m => n(at + t * beat, m, 2 * beat, .06, 'brass')));
      fx.push({type:'duck', bus:'music', time:at, hold:2 * bar - .3, release:1.2, gain:.35});
    } else if (name === 'secret'){
      ladder(pcs, 72 + r, 96 + r).slice(0, 10).forEach((m, i) => n(at + i * s16 / 2, m, .7, .06 + .006 * i, 'glass'));
    } else if (name === 'boss'){
      n(at, 36 + r, .9, .34, 'thud'); n(at, 48 + r, 1.4, .08, 'brass'); n(at, 55 + r, 1.4, .07, 'brass');
      fx.push({type:'boss', on:true, time:at});
    } else if (name === 'boss_end'){
      n(at, 72 + r, 1.2, .1); n(at + s16, 79 + r, 1.2, .08);
      fx.push({type:'boss', on:false, time:at});
    } else if (name === 'transition'){
      // A riser that lands on the boundary `time`; variants follow the jungle stinger roles.
      const len = Math.max(.25, a.lead || bar), v = a.variant || 'riser', t0 = at - len;
      n(t0, v === 'dark' ? 36 + r : 48 + r, len, v === 'dark' ? .07 : .09, 'swell');
      if (v === 'pulse') for (let i = 0; i < 4; i++) n(at - (4 - i) * s16, 84 + r, .05, .02 + .01 * i, 'tick');
      if (v === 'shimmer') ladder(pcs, 72 + r, 96 + r).slice(0, 8).forEach((m, i) => n(at - beat / 2 + i * s16 / 4, m, .5, .04, 'glass'));
      if (v === 'dark') n(at, 36 + r, .9, .2, 'thud');
      ladder(pcs, 62 + r, 86 + r, 3).slice(0, 3).forEach((m, i) => n(at + i * .02, m, 2.4, .06));
    } else throw new TypeError('Unknown event ' + name);
    return {name, at, notes:notes.filter(x => x.time >= now - 1e-9), fx, chain};
  }

  // ---------------------------------------------------------------- Web Audio voices
  const noiseCache = new WeakMap();
  function noise(ctx){
    if (!noiseCache.has(ctx)){
      const b = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate), d = b.getChannelData(0);
      let s = 12345; for (let i = 0; i < d.length; i++){ s = (Math.imul(s, 1664525) + 1013904223) >>> 0; d[i] = s / 2147483648 - 1; }
      noiseCache.set(ctx, b);
    }
    return noiseCache.get(ctx);
  }
  const hz = m => 440 * 2 ** ((m - 69) / 12);
  function env(param, t, peak, att, dur){
    param.setValueAtTime(.0001, t); param.linearRampToValueAtTime(peak, t + att);
    param.exponentialRampToValueAtTime(.0001, t + Math.max(att + .01, dur));
  }
  function osc(ctx, type, f, t, end, out){
    const o = ctx.createOscillator(); o.type = type; o.frequency.setValueAtTime(f, t); o.connect(out); o.start(t); o.stop(end); return o;
  }
  function voice(ctx, dest, x, scale){
    const t = Math.max(x.time, ctx.currentTime), f = hz(x.midi), g = ctx.createGain(), peak = x.gain * scale, end = t + x.dur + .1;
    g.connect(dest);
    const partial = (ratio, amp, decay, type = 'sine') => { const pg = ctx.createGain(); pg.connect(g); env(pg.gain, t, amp, .002, decay); osc(ctx, type, f * ratio, t, end, pg); };
    if (x.voice === 'glass'){
      // two-operator FM bell
      const mod = ctx.createOscillator(), mg = ctx.createGain(), car = ctx.createOscillator();
      mod.frequency.setValueAtTime(f * 3.5, t); mg.gain.setValueAtTime(f * 2.2, t); mg.gain.exponentialRampToValueAtTime(f * .05, t + x.dur);
      car.frequency.setValueAtTime(f, t); mod.connect(mg); mg.connect(car.frequency); car.connect(g);
      env(g.gain, t, peak, .003, x.dur); mod.start(t); car.start(t); mod.stop(end); car.stop(end);
    } else if (x.voice === 'brass'){
      const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 2; lp.connect(g);
      lp.frequency.setValueAtTime(500, t); lp.frequency.linearRampToValueAtTime(2600, t + .06); lp.frequency.exponentialRampToValueAtTime(1100, t + Math.max(.1, x.dur));
      osc(ctx, 'sawtooth', f, t, end, lp); osc(ctx, 'sawtooth', f * 1.004, t, end, lp);
      g.gain.setValueAtTime(.0001, t); g.gain.linearRampToValueAtTime(peak * .5, t + .03);
      g.gain.setValueAtTime(peak * .5, t + Math.max(.04, x.dur - .08)); g.gain.exponentialRampToValueAtTime(.0001, t + x.dur + .08);
    } else if (x.voice === 'thud'){
      const o = osc(ctx, 'sine', f * 2.5, t, end, g); o.frequency.exponentialRampToValueAtTime(f, t + .08); env(g.gain, t, peak, .004, x.dur);
    } else if (x.voice === 'tick' || x.voice === 'buzz'){
      const lp = ctx.createBiquadFilter(); lp.type = x.voice === 'tick' ? 'bandpass' : 'lowpass'; lp.frequency.value = x.voice === 'tick' ? f : 900; lp.connect(g);
      osc(ctx, x.voice === 'tick' ? 'triangle' : 'sawtooth', f, t, end, lp); env(g.gain, t, peak, .002, x.dur);
    } else if (x.voice === 'swell'){
      const s = ctx.createBufferSource(), bp = ctx.createBiquadFilter(); s.buffer = noise(ctx); s.loop = true;
      bp.type = 'bandpass'; bp.Q.value = 1.4; bp.frequency.setValueAtTime(f * 2, t); bp.frequency.exponentialRampToValueAtTime(Math.min(9000, f * 64), t + x.dur);
      s.connect(bp); bp.connect(g); g.gain.setValueAtTime(.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + x.dur);
      g.gain.exponentialRampToValueAtTime(.0001, t + x.dur + .12); s.start(t); s.stop(t + x.dur + .2);
    } else {
      g.gain.value = peak;
      if (x.voice === 'steel'){ partial(1, 1, x.dur); partial(2.76, .45, .12); partial(4.1, .15, .05); }
      else if (x.voice === 'mallet'){ partial(1, 1, x.dur * .8); partial(3.99, .35, .08); }
      else { partial(1, 1, x.dur); partial(5.9, .3, .05); partial(3, .08, .2, 'triangle'); } // kalimba
    }
    return g;
  }
  function play(ctx, dest, notes, scale = 1){ for (const x of notes) voice(ctx, dest, x, scale); return notes.length; }
  return {NAMES, parse, key, palette, ladder, schedule, play};
})();
