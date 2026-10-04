/* KAPOK in the player: linear playback of the suite with a live score view.
   Data: kapok/kapok.json (written by src/tracks/kapok_web.py). Audio: audio/kapok.m4a,
   or the same bytes as base64 chunks (audio/kapok_b64/*) when the binary isn't checked in. */
'use strict';
(() => {
  const qs = s => document.querySelector(s);
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const fmt = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  const SOUNDTRACKS = [['jungle','Jungle'],['space','Space'],['canopy','Jungle Canopy Level'],['mine','Mine Cart Level'],
    ['underwater','Underwater Level'],['orbital','Orbital Rainforest'],['crystal','Crystal Caverns'],
    ['greenhouse','Clockwork Greenhouse'],['kapok','KAPOK (Oldfield-style suite)']];
  const FAM = {mallet:'#E8A33A', metal:'#D9C36A', bell:'#F4E3A1', drum:'#D8643B', perc:'#C98C5A', pluck:'#9BCB6B',
    guitar:'#6FBF73', bass:'#3E9E6E', bowed:'#C77DB3', keys:'#7FA8E0', organ:'#8E8BD8', flute:'#62D0C6',
    reed:'#4FB0A8', brass:'#E0B04A', voice:'#F09AA7', creature:'#B5E061', synth:'#9C7BF0'};
  const MOVE_SUB = ['the climb begins at the roots', 'the creatures become the band', 'rain on the leaves, 3/4',
    'every instrument not yet heard takes a bow', 'Ommadawn drums, 12/8', 'one by one … plus tubular bells',
    'the canopy opens into the stars', 'a reel that will not stop speeding up'];
  const sel = qs('#soundscape');
  for (const [v, n] of SOUNDTRACKS){ const o = document.createElement('option'); o.value = v; o.textContent = n; sel.appendChild(o); }
  sel.value = 'kapok';
  sel.addEventListener('change', () => { if (sel.value !== 'kapok') location.href = 'index.html?soundscape=' + sel.value; });

  let D = null, ctx = null, audio = null, music = null, duck = null, master = null, mcBuf = null, ready = false;
  let mcOn = true, mcSources = [], mcDone = new Set(), lastBar = -1, lastMove = -1, chain = null;
  const listeners = {bar:new Set(), movement:new Set()};
  const status = t => qs('#status').textContent = t;

  // ------------------------------------------------------------------ loading
  async function bytes(file, chunks){
    try {
      const r = await fetch(file);
      if (r.ok){ const b = await r.arrayBuffer(); if (b.byteLength > 1000 && !/^<|^version https/.test(new TextDecoder().decode(b.slice(0, 40)))) return b; }
    } catch (e) {}
    let done = 0;
    const parts = await Promise.all(chunks.map(c => fetch(c).then(r => { if (!r.ok) throw new Error('Could not load ' + c); return r.text(); })
      .then(t => { status(`Loading audio… ${Math.round(100 * ++done / chunks.length)}%`); return t; })));
    const bin = atob(parts.join('')), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }
  async function load(){
    D = await (await fetch('kapok/kapok.json')).json();
    qs('#ninst').textContent = D.credits.length;
    buildUI();
    status('Loading audio…');
    const buf = await bytes(D.audio.file, D.audio.b64);
    audio = new Audio(); audio.preload = 'auto'; audio.crossOrigin = 'anonymous';
    audio.src = URL.createObjectURL(new Blob([buf], {type:'audio/mp4'}));
    audio.addEventListener('ended', () => setPlaying(false));
    if (D.audio.mc_file || (D.audio.mc_b64 || []).length) {
      try { D._mcBytes = await bytes(D.audio.mc_file, D.audio.mc_b64); } catch (e) { console.warn(e); }
    }
    ready = true; status(`Ready · ${fmt(D.length)} · press play`);
  }
  function initAudio(){
    ctx = new AudioContext({latencyHint:'playback'});
    music = ctx.createMediaElementSource(audio);
    duck = ctx.createGain(); master = ctx.createGain();
    const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 20000; E.lp = lp;
    music.connect(duck); duck.connect(lp); lp.connect(master); master.connect(ctx.destination);
    E.voice = ctx.createGain(); E.voice.gain.value = 1.1; E.voice.connect(master);
    E.events = ctx.createGain(); E.events.gain.value = .9; E.events.connect(master);
    if (D._mcBytes) ctx.decodeAudioData(D._mcBytes.slice(0)).then(b => mcBuf = b).catch(e => console.warn(e));
  }
  const E = {};

  // ------------------------------------------------------------------ time helpers
  const now = () => audio ? audio.currentTime : 0;
  function idx(list, t){ let lo = 0, hi = list.length - 1; while (lo < hi){ const m = (lo + hi + 1) >> 1; if (list[m].t <= t) lo = m; else hi = m - 1; } return lo; }
  function beatsIn(meter){ const [n, d] = meter.split('/').map(Number); return n * 4 / d; }
  function grid(t){
    const i = idx(D.bars, t), b = D.bars[i], nb = D.bars[i + 1] || {t:b.t + (b.t - (D.bars[i - 1] || b).t)};
    const beats = beatsIn(b.meter), beat = (nb.t - b.t) / beats;
    const m = idx(D.movements, t);
    const first = D.bars.findIndex(x => x.t >= D.movements[m].t - 1e-3);
    return {i, bar:b, beat, bpm:60 / beat, beats, mov:m, barInMove:i - Math.max(0, first) + 1};
  }

  // ------------------------------------------------------------------ transport
  function setPlaying(on){
    qs('#playIcon').setAttribute('d', on ? 'M7 5h4v14H7zM13 5h4v14h-4z' : 'M8 5v14l11-7z');
    qs('#playLabel').textContent = on ? 'Pause' : 'Play';
  }
  async function toggle(){
    if (!ready) return;
    if (!ctx) initAudio();
    await ctx.resume();
    if (audio.paused){ await audio.play(); setPlaying(true); status('Playing'); }
    else { audio.pause(); stopMC(); setPlaying(false); status('Paused'); }
  }
  function seek(t){
    if (!audio) return;
    audio.currentTime = clamp(t, 0, D.length - .05); stopMC();
    mcDone = new Set(D.mc.filter(m => m.at + m.dur < t).map(m => m.at));
  }
  qs('#play').addEventListener('click', toggle);
  document.addEventListener('keydown', e => { if (e.code === 'Space' && e.target === document.body){ e.preventDefault(); toggle(); } });

  // ------------------------------------------------------------------ announcer (scheduled on the audio clock)
  function stopMC(){ for (const s of mcSources) try { s.stop(); } catch (e) {} mcSources = []; if (duck) duck.gain.setTargetAtTime(1, ctx.currentTime, .1); }
  function scheduleMC(){
    if (!mcOn || !mcBuf || !ctx || audio.paused) return;
    const t = now();
    for (const m of D.mc){
      if (mcDone.has(m.at) || m.at > t + .35 || m.at + m.dur < t) continue;
      mcDone.add(m.at);
      const when = ctx.currentTime + Math.max(0, m.at - t), off = m.offset + Math.max(0, t - m.at);
      const s = ctx.createBufferSource(); s.buffer = mcBuf; s.connect(E.voice);
      s.start(when, off, Math.max(.05, m.dur - Math.max(0, t - m.at)));
      mcSources.push(s);
      duck.gain.setTargetAtTime(.6, when - .1, .08); duck.gain.setTargetAtTime(1, when + m.dur, .25);
      const a = qs('#announce'); setTimeout(() => { a.textContent = m.text; a.classList.add('on'); }, Math.max(0, (m.at - t) * 1000));
      setTimeout(() => a.classList.remove('on'), Math.max(0, (m.at - t + m.dur + 1.2) * 1000));
    }
  }
  qs('#mcToggle').addEventListener('click', e => {
    mcOn = !mcOn; e.currentTarget.setAttribute('aria-pressed', mcOn); e.currentTarget.textContent = mcOn ? 'Announcer on' : 'Announcer off';
    if (!mcOn) stopMC();
  });

  // ------------------------------------------------------------------ game events, in key and on Kapok's own grid
  function event(name, opts = {}){
    if (!ctx || !audio || audio.paused) return null;
    const t = now(), g = grid(t), offset = ctx.currentTime - t;
    const r = CanopyEvents.schedule(name, {bpm:g.bpm, beatsPerBar:g.beats, now:ctx.currentTime, origin:g.bar.t + offset,
      chord:g.bar.chord || 'Dm', key:'D major', chain, palette:{lead:'kalimba'}, opts});
    chain = r.chain;
    CanopyEvents.play(ctx, E.events, r.notes, 1);
    for (const f of r.fx){
      if (f.type === 'duck' || f.type === 'fade'){
        duck.gain.setTargetAtTime(f.gain || .4, f.time, .03);
        duck.gain.setTargetAtTime(1, f.time + (f.hold || f.dur || .4), f.release || .4);
      }
    }
    return r;
  }
  document.querySelectorAll('[data-ev]').forEach(b => b.addEventListener('click', () => event(b.dataset.ev)));

  window.CanopyGame = Object.freeze({
    mode:'kapok', event, seek, play:() => audio && audio.paused && toggle(), pause:() => audio && !audio.paused && toggle(),
    getState:() => { if (!D || !audio) return {ready:false}; const g = grid(now());
      return {ready:true, time:now(), length:D.length, movement:D.movements[g.mov].name, bar:g.barInMove, meter:g.bar.meter, chord:g.bar.chord, bpm:g.bpm}; },
    on:(n, fn) => { if (!listeners[n]) throw new TypeError('Unknown event ' + n); listeners[n].add(fn); return () => listeners[n].delete(fn); },
    off:(n, fn) => listeners[n] && listeners[n].delete(fn),
  });
  const emit = (n, x) => listeners[n].forEach(fn => { try { fn(x); } catch (e) { console.warn(e); } });

  // ------------------------------------------------------------------ UI
  const MOVE_COL = ['#2f6b4f','#4f8a3c','#456b86','#a07a35','#9a4a2e','#c08a2a','#2b2d63','#b0623a'];
  function buildUI(){
    const tl = qs('#timeline'), mv = qs('#moves');
    D.movements.forEach((m, i) => {
      const end = (D.movements[i + 1] || {t:D.length}).t, seg = document.createElement('div');
      seg.className = 'seg'; seg.style.left = (100 * m.t / D.length) + '%'; seg.style.width = (100 * (end - m.t) / D.length) + '%';
      seg.style.background = MOVE_COL[i % MOVE_COL.length]; seg.textContent = m.name.replace(/^Interlude: /, ''); tl.appendChild(seg);
      const c = document.createElement('button'); c.className = 'chip'; c.type = 'button'; c.textContent = m.name;
      c.addEventListener('click', () => seek(m.t + .01)); mv.appendChild(c);
    });
    const head = document.createElement('div'); head.className = 'head'; head.id = 'head'; tl.appendChild(head);
    tl.addEventListener('click', e => { const r = tl.getBoundingClientRect(); seek((e.clientX - r.left) / r.width * D.length); });
    tl.addEventListener('keydown', e => { if (e.key === 'ArrowRight') seek(now() + 10); if (e.key === 'ArrowLeft') seek(now() - 10); });
    qs('#tEnd').textContent = fmt(D.length);
    // family board: one heat strip per family over the whole piece
    const board = qs('#board');
    for (const f of D.families){
      const lab = document.createElement('div'); lab.textContent = f; board.appendChild(lab);
      const cv = document.createElement('canvas'), a = D.activity[f]; cv.width = 600; cv.height = 12; cv.dataset.f = f;
      const g = cv.getContext('2d');
      for (let x = 0; x < 600; x++){
        const i0 = Math.floor(x / 600 * a.length), i1 = Math.max(i0 + 1, Math.floor((x + 1) / 600 * a.length));
        let v = 0; for (let i = i0; i < i1; i++) v = Math.max(v, +a[i]);
        if (v){ g.globalAlpha = .25 + .75 * Math.min(1, v / 5); g.fillStyle = FAM[f] || '#45C2BA'; g.fillRect(x, 1, 1, 10); }
      }
      cv.addEventListener('click', e => { const r = cv.getBoundingClientRect(); seek((e.clientX - r.left) / r.width * D.length); });
      board.appendChild(cv);
    }
    const cr = qs('#credits');
    D.credits.forEach((c, i) => {
      const el = document.createElement('div'); el.className = 'credit'; el.id = 'cr' + i;
      el.innerHTML = `<time>${fmt(c.t)}</time><div><div class="n" style="color:${FAM[c.family] || 'inherit'}">${pretty(c.id)}</div><div class="d">${c.desc}</div></div>`;
      el.addEventListener('click', () => seek(c.t - .3)); cr.appendChild(el);
    });
  }
  const pretty = id => id.split('.')[1].replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());

  let lastCredit = -1, nowKey = '';
  function updateUI(t){
    const g = grid(t), m = D.movements[g.mov];
    qs('#head').style.left = (100 * t / D.length) + '%';
    qs('#tNow').textContent = fmt(t); qs('#hudTime').textContent = `${fmt(t)} / ${fmt(D.length)}`;
    qs('#hudMove').textContent = m.name; qs('#hudSub').textContent = MOVE_SUB[g.mov] || '';
    qs('#hudBar').textContent = `Bar ${g.barInMove} · ${g.bar.meter} · ${Math.round(g.bpm)} BPM`;
    qs('#hudChord').textContent = g.bar.chord || '';
    if (g.i !== lastBar){ lastBar = g.i; emit('bar', {bar:g.barInMove, meter:g.bar.meter, chord:g.bar.chord, time:t}); }
    if (g.mov !== lastMove){ lastMove = g.mov; emit('movement', {index:g.mov, name:m.name, time:t});
      document.querySelectorAll('#moves .chip').forEach((c, i) => c.setAttribute('aria-pressed', i === g.mov)); }
    // credits & "now entering"
    const k = D.credits.findIndex(c => c.t > t + .05), latest = (k < 0 ? D.credits.length : k) - 1;
    if (latest !== lastCredit){
      document.querySelectorAll('.credit').forEach((el, i) => { el.classList.toggle('in', i <= latest); el.classList.toggle('latest', i === latest); });
      if (latest >= 0){ const el = qs('#cr' + latest); el.parentNode.scrollTop = el.offsetTop - el.parentNode.offsetTop - 120; }
      lastCredit = latest;
    }
    const c = D.credits[latest];
    qs('#hudEnter').innerHTML = c && t - c.t < 6 ? `Now entering<br><b>${pretty(c.id)}</b><small>${c.desc}</small>` : '';
    // who is sounding
    const on = [];
    for (const id in D.spans){ const s = D.spans[id]; let lo = 0, hi = s.length - 1, hit = false;
      while (lo <= hi){ const mid = (lo + hi) >> 1; if (s[mid][1] < t) lo = mid + 1; else if (s[mid][0] > t) hi = mid - 1; else { hit = true; break; } }
      if (hit) on.push(id); }
    const key = on.join();
    if (key !== nowKey){
      nowKey = key; const box = qs('#now'); box.innerHTML = '';
      on.sort((a, b) => a.localeCompare(b)).forEach(id => { const s = document.createElement('span'); const fam = id.split('.')[0];
        s.style.setProperty('--c', FAM[fam] || '#45C2BA'); s.textContent = pretty(id);
        const cr = D.credits.find(x => x.id === id); if (cr && t - cr.t < 6) s.className = 'new'; box.appendChild(s); });
      qs('#count').textContent = on.length;
    }
    return {g, on};
  }

  // ------------------------------------------------------------------ the scene: climbing the kapok
  const cv = qs('#scene'), G = cv.getContext('2d');
  const W = 320, H = 140, WORLD = 1400;
  function mulberry(a){ return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const SKY = [['#0e2a1f','#1d4a32'],['#2f6b3f','#7fb36a'],['#3a5466','#7b93a3'],['#a4783a','#e8c27a'],['#3b1f2a','#b4583a'],
    ['#4c8fc4','#cfe6f2'],['#05061a','#1b1f4d'],['#e07a4f','#ffd59a']];
  const leaves = []; { const r = mulberry(7); for (let i = 0; i < 70; i++) leaves.push({y:200 + r() * 950, x:r() * 320, s:6 + r() * 16, k:r()}); }
  const stars = []; { const r = mulberry(9); for (let i = 0; i < 120; i++) stars.push({x:r() * 320, y:r() * 140, k:r()}); }
  const parts = [];
  const mix = (a, b, x) => { const p = s => [1, 3, 5].map(i => parseInt(s.slice(i, i + 2), 16)); const A = p(a), B = p(b);
    return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * x)).join(',')})`; };
  function frame(t, info){
    const p = clamp(t / D.length), g = info ? info.g : {mov:0, beat:.5, bar:{t:0}};
    const camY = (WORLD - H) * (1 - Math.pow(p, .9));   // 0 = top of the world (sky)
    const m = g.mov, nxt = Math.min(SKY.length - 1, m + 1), mt = D.movements[m].t, me = (D.movements[m + 1] || {t:D.length}).t;
    const f = clamp((t - mt) / Math.max(1, me - mt) * 3 - 2);    // blend into the next sky in the last third
    const sky = G.createLinearGradient(0, 0, 0, H);
    sky.addColorStop(0, mix(SKY[m][0], SKY[nxt][0], f)); sky.addColorStop(1, mix(SKY[m][1], SKY[nxt][1], f));
    G.fillStyle = sky; G.fillRect(0, 0, W, H);
    if (m >= 6 || p > .78){ G.globalAlpha = m >= 6 ? 1 : (p - .78) * 4; for (const s of stars){ G.fillStyle = s.k > .9 ? '#fff6d0' : '#c8d6ff';
      G.globalAlpha *= 1; G.fillRect(s.x | 0, s.y | 0, 1, 1); } G.globalAlpha = 1; }
    // trunk
    const beatPh = ((t - g.bar.t) / g.beat) % 1, pulse = Math.exp(-beatPh * 5);
    for (let y = 0; y < H; y++){
      const wy = camY + y, w = 18 + 26 * Math.pow(clamp(wy / WORLD), 1.6) + (wy > WORLD - 60 ? (wy - (WORLD - 60)) * 1.2 : 0);
      if (wy < 120) continue;
      G.fillStyle = (wy | 0) % 9 < 2 ? '#4a3524' : '#5b4330'; G.fillRect(160 - w / 2 | 0, y, w | 0, 1);
      G.fillStyle = '#6e5539'; G.fillRect(160 - w / 2 + 3 | 0, y, 2, 1);
    }
    // branches and leaf clusters (parallax)
    for (const l of leaves){
      const y = l.y - camY; if (y < -30 || y > H + 30) continue;
      G.fillStyle = '#4a3524'; G.fillRect(l.x < 160 ? l.x | 0 : 160, y + l.s / 2 | 0, Math.abs(160 - l.x) | 0, 2);
      G.fillStyle = l.k > .5 ? '#2f7a3f' : '#3f9a4a'; G.fillRect(l.x - l.s / 2 | 0, y | 0, l.s | 0, l.s / 2 | 0);
      G.fillStyle = '#5cbf5a'; G.fillRect(l.x - l.s / 3 | 0, y | 0, l.s / 2 | 0, 2);
    }
    // ground near the start
    const gy = WORLD - camY - 10; if (gy < H){ G.fillStyle = '#1c3a22'; G.fillRect(0, gy | 0, W, H); G.fillStyle = '#2d5a31';
      for (let x = 0; x < W; x += 6) G.fillRect(x, gy - 3 - (x * 7 % 5) | 0, 3, 4); }
    // movement weather
    if (m === 2){ G.fillStyle = 'rgba(190,215,235,.55)'; for (let i = 0; i < 70; i++){ const x = (i * 47 + t * 30) % W, y = (i * 23 + t * 140) % H; G.fillRect(x | 0, y | 0, 1, 3); } }
    if (m === 7){ for (let i = 0; i < 3; i++){ const x = (t * (40 + i * 13) + i * 110) % (W + 40) - 20, y = 30 + i * 22 + Math.sin(t * 3 + i) * 4;
      G.fillStyle = '#1b1b1b'; G.fillRect(x | 0, y | 0, 9, 3); G.fillStyle = '#f2c94c'; G.fillRect(x + 8 | 0, y | 0, 4, 2);
      G.fillStyle = '#1b1b1b'; G.fillRect(x + 2 | 0, y - (Math.sin(t * 14 + i) > 0 ? 3 : -2) | 0, 5, 2); } }
    // notes rise from the tree for every family that is playing (colour = family)
    if (info){
      for (const id of info.on){ if (Math.random() < .035){ const fam = id.split('.')[0];
        parts.push({x:150 + (Math.random() - .5) * 120, y:H - 6 - Math.random() * 60, vy:-(10 + Math.random() * 18), c:FAM[fam] || '#fff', life:2.2}); } }
      const drums = info.on.filter(id => /^(drum|perc)\./.test(id)).length;
      if (drums && pulse > .2){ G.fillStyle = `rgba(255,170,90,${.10 * pulse * Math.min(1, drums / 4)})`; G.fillRect(0, 0, W, H); }
    }
    const dt = 1 / 60;
    for (let i = parts.length - 1; i >= 0; i--){ const q = parts[i]; q.y += q.vy * dt; q.x += Math.sin(q.y * .1) * .3; q.life -= dt;
      if (q.life <= 0){ parts.splice(i, 1); continue; }
      G.globalAlpha = Math.min(1, q.life); G.fillStyle = q.c; G.fillRect(q.x | 0, q.y | 0, 2, 2); G.fillRect((q.x | 0) + 2, (q.y | 0) - 3, 1, 4); }
    G.globalAlpha = 1;
    // the climber, bobbing on the beat
    const cy = 92 - (pulse * 2 | 0), cx = 160 + 13 + Math.round(Math.sin(t * .7) * 2);
    G.fillStyle = '#e8d4a8'; G.fillRect(cx, cy, 4, 4); G.fillStyle = '#c0392b'; G.fillRect(cx - 1, cy + 4, 6, 5);
    G.fillStyle = '#2c3e50'; G.fillRect(cx, cy + 9, 2, 3); G.fillRect(cx + 3, cy + 9, 2, 3);
  }

  function loop(){
    if (D){
      const t = now(), info = audio ? updateUI(t) : null;
      frame(t, info); scheduleMC();
    }
    requestAnimationFrame(loop);
  }
  load().catch(e => { console.warn(e); status('Could not load KAPOK: ' + e.message + ' (serve the player folder over http)'); });
  requestAnimationFrame(loop);
})();
