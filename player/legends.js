/* Glimmer Legends: a Rayman Legends–inspired soundpack (src/tracks/legends.py).
   Worlds: three phase-locked layer loops per world; layer changes land on the bar line.
   Glims: each pickup plays the next chord tone of the current bar (1/16-quantised).
   Musical level: the beatmap drives the runner; glims play on their beats. */
'use strict';
(() => {
  const qs = s => document.querySelector(s), qsa = s => [...document.querySelectorAll(s)];
  const SOUNDTRACKS = [['jungle','Jungle'],['space','Space'],['canopy','Jungle Canopy Level'],['mine','Mine Cart Level'],
    ['underwater','Underwater Level'],['orbital','Orbital Rainforest'],['crystal','Crystal Caverns'],
    ['greenhouse','Clockwork Greenhouse'],['kapok','KAPOK (Oldfield-style suite)'],['microgame','Microgame Rush (WarioWare-style)'],
    ['legends','Glimmer Legends (Rayman Legends-style)']];
  const PAGES = {kapok:'kapok.html', microgame:'microgame.html', legends:'legends.html'};
  const sel = qs('#soundscape');
  for (const [v, n] of SOUNDTRACKS){ const o = document.createElement('option'); o.value = v; o.textContent = n; sel.appendChild(o); }
  sel.value = 'legends';
  sel.addEventListener('change', () => { location.href = PAGES[sel.value] || 'index.html?soundscape=' + sel.value; });

  let D = null, ctx = null;
  const B = {};                 // decoded sprites: id -> AudioBuffer slices
  const E = {};
  const listeners = {bar:new Set(), action:new Set(), judgement:new Set(), section:new Set()};
  const emit = (n, x) => listeners[n].forEach(fn => { try { fn(x); } catch (e) { console.warn(e); } });
  const hint = t => qs('#hint').textContent = t;

  // ------------------------------------------------------------------ loading (sync impulse at sample 1000)
  async function bytes(file, b64){
    try { const r = await fetch(file); if (r.ok){ const b = await r.arrayBuffer(); if (b.byteLength > 1000 && !/^<|^version https/.test(new TextDecoder().decode(b.slice(0, 40)))) return b; } } catch (e) {}
    const t = await (await fetch(b64)).text(), bin = atob(t), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }
  function cut(buf, base, k, start, len){
    const n = Math.max(1, Math.round(len * k)), out = ctx.createBuffer(2, n, ctx.sampleRate), s = base + Math.round(start * k);
    for (let c = 0; c < 2; c++) out.copyToChannel(buf.getChannelData(Math.min(c, buf.numberOfChannels - 1)).subarray(s, s + n), c);
    return out;
  }
  const loading = {};
  function sprite(id){
    if (!loading[id]) loading[id] = (async () => {
      const sp = D.sprites[id], buf = await ctx.decodeAudioData(await bytes(sp.file, sp.b64));
      const k = ctx.sampleRate / D.sr, d = buf.getChannelData(0);
      let p = 0, mx = 0; for (let j = 0; j < Math.round(7000 * k); j++){ const v = Math.abs(d[j]); if (v > mx){ mx = v; p = j; } }
      return {buf, base:p - Math.round(sp.sync * k), k};
    })();
    return loading[id];
  }
  async function loadWorld(w){
    if (B[w.id]) return B[w.id];
    const s = await sprite(w.id), out = {stems:{}, tails:{}, stingers:{}};
    for (const [t, o] of Object.entries(w.stems)){ out.stems[t] = cut(s.buf, s.base, s.k, o.start, o.length); out.tails[t] = cut(s.buf, s.base, s.k, o.tail, o.tail_length); }
    for (const [n, o] of Object.entries(w.stingers)) out.stingers[n] = cut(s.buf, s.base, s.k, o.start, o.length);
    return (B[w.id] = out);
  }
  async function loadGlims(){
    if (B.glims) return B.glims;
    const s = await sprite('glims'), out = {};
    for (const [n, o] of Object.entries(D.glims)) out[n] = cut(s.buf, s.base, s.k, o.start, o.length);
    return (B.glims = out);
  }
  async function loadLevel(){
    if (B.level) return B.level;
    const s = await sprite('level'), o = D.level.audio;
    return (B.level = cut(s.buf, s.base, s.k, o.start, o.length));
  }
  function src(buf, dest, t, {loop = false, gain = 1} = {}){
    const s = ctx.createBufferSource(), g = ctx.createGain(); s.buffer = buf; s.loop = loop; g.gain.value = gain;
    s.connect(g); g.connect(dest); s.start(t); return {s, g};
  }

  // ------------------------------------------------------------------ worlds
  const Wd = {world:null, playing:false, t0:0, layer:3, srcs:{}, king:false, kingUntil:0, chain:null, auto:false, banner:null};
  const bar = w => w.beats_per_bar * 60 / w.bpm;
  const nextBar = w => { const b = bar(w), n = Math.ceil((ctx.currentTime + .05 - Wd.t0) / b - 1e-9); return Wd.t0 + n * b; };
  const barIndex = w => Math.floor(((ctx.currentTime - Wd.t0) / bar(w)) % w.bars + w.bars) % w.bars;
  async function playWorld(w, at){
    const A = await loadWorld(w); loadGlims();
    at = at ?? ctx.currentTime + .1;
    Wd.world = w; Wd.t0 = at; Wd.playing = true; Wd.srcs = {};
    for (const t of [1, 2, 3]){
      const on = t <= Wd.layer ? 1 : 0, x = src(A.stems['tier' + t], E.music, at, {loop:true, gain:on});
      Wd.srcs[t] = x;
    }
    qs('#worldPlay').textContent = 'Stop';
    hint(`${w.name} · ${w.key} · ${w.meter} · ${w.bpm} BPM — G collects a glim`);
  }
  function stopWorld(at){
    if (!Wd.playing) return;
    const w = Wd.world, A = B[w.id];
    at = at ?? ctx.currentTime + .03;
    for (const [t, x] of Object.entries(Wd.srcs)){
      const lvl = x.g.gain.value;
      x.s.stop(at);
      if (lvl > .01) src(A.tails['tier' + t], E.music, at, {gain:lvl});
    }
    Wd.playing = false; Wd.srcs = {}; qs('#worldPlay').textContent = 'Play';
  }
  function setLayer(n){
    Wd.layer = Math.max(1, Math.min(3, n));
    qsa('[data-tier]').forEach(b => b.setAttribute('aria-pressed', +b.dataset.tier === Wd.layer));
    if (!Wd.playing) return;
    const tb = nextBar(Wd.world);
    for (const t of [1, 2, 3]){
      const g = Wd.srcs[t].g.gain, want = t <= Wd.layer ? 1 : 0;
      g.cancelScheduledValues(tb - .02); g.setValueAtTime(g.value, tb - .02); g.linearRampToValueAtTime(want, tb + (want ? .02 : .25));
    }
  }
  async function switchWorld(w){
    qsa('[data-world]').forEach(b => b.setAttribute('aria-pressed', b.dataset.world === w.id));
    if (!Wd.playing){ Wd.world = w; drawBanner(w.name); return; }
    await loadWorld(w);
    const tb = nextBar(Wd.world);
    stopWorld(tb); await playWorld(w, tb); drawBanner(w.name);
  }
  function drawBanner(text){ Wd.banner = {text, t:ctx ? ctx.currentTime : 0}; }
  function collect(){
    if (!Wd.playing || !B.glims) return null;
    const w = Wd.world, beat = 60 / w.bpm, step = beat / 4;
    const at = Wd.t0 + Math.ceil((ctx.currentTime + .012 - Wd.t0) / step) * step;
    const b = Math.floor(((at - Wd.t0) / bar(w)) % w.bars), tones = w.chord_tones[b];
    const reset = !Wd.chain || ctx.currentTime - Wd.chain.at > 1.2;
    const idx = reset ? 0 : Math.min(tones.length - 1, Wd.chain.idx + 1);
    const midi = tones[idx], king = Wd.king && ctx.currentTime < Wd.kingUntil;
    const gb = B.glims[(king ? 'king_' : 'glim_') + midi]; if (gb) src(gb, E.sfx, at);
    Wd.chain = {idx, at:ctx.currentTime, bar:b};
    Glims.burst(king);
    return {midi, king, time:at};
  }
  function king(on = true){ Wd.king = on; Wd.kingUntil = on ? (ctx ? ctx.currentTime : 0) + 12 : 0; qs('#kingBtn').setAttribute('aria-pressed', on); }
  async function sting(name){
    const w = Wd.world; if (!w) return;
    const A = await loadWorld(w), now = ctx.currentTime;
    if (name === 'death'){
      src(A.stingers.death, E.sfx, now + .02);
      E.lp.frequency.setTargetAtTime(500, now, .05); E.music.gain.setTargetAtTime(.35, now, .05);
      E.lp.frequency.setTargetAtTime(20000, now + 2.4, .4); E.music.gain.setTargetAtTime(1, now + 2.4, .4);
      Glims.flash('#ff4f4f'); drawBanner('Ouch!');
    } else if (name === 'complete'){
      const tb = Wd.playing ? nextBar(w) : now + .05;
      if (Wd.playing){ E.music.gain.setTargetAtTime(.0001, tb - .3, .12); setTimeout(() => { stopWorld(); E.music.gain.setValueAtTime(1, ctx.currentTime); }, (tb - now + 1) * 1000); }
      src(A.stingers.complete, E.sfx, tb); drawBanner('Level complete!');
    } else {
      const beat = 60 / w.bpm, at = Wd.playing ? Wd.t0 + Math.ceil((now + .02 - Wd.t0) / beat) * beat : now + .02;
      if (Wd.playing) { E.music.gain.setTargetAtTime(.55, at, .03); E.music.gain.setTargetAtTime(1, at + 1.2, .3); }
      src(A.stingers[name], E.sfx, at);
      if (name === 'king'){ king(true); Glims.flash('#ffd23f'); }
      drawBanner({checkpoint:'Checkpoint!', freed:'Teensies freed!', king:'King glims!'}[name]);
    }
  }

  // ------------------------------------------------------------------ the musical level
  const Lv = {on:false, auto:true, t0:0, idx:0, hits:{perfect:0, good:0, miss:0}, glims:0, judged:new Set(), pops:[], lastSection:-1, src:null, lastBar:-1};
  async function startLevel({auto = true} = {}){
    if (Wd.playing) stopWorld();
    stopLevel();
    const buf = await loadLevel(); await loadGlims();
    Object.assign(Lv, {on:true, auto, idx:0, hits:{perfect:0, good:0, miss:0}, glims:0, judged:new Set(), pops:[], lastSection:-1, lastBar:-1});
    Lv.t0 = ctx.currentTime + .3;
    const me = Lv.src = src(buf, E.music, Lv.t0);
    me.s.onended = () => { if (Lv.src === me && Lv.on){ Lv.on = false; hint(`Finished — Perfect ${Lv.hits.perfect} · Good ${Lv.hits.good} · Miss ${Lv.hits.miss}`); } };
    // glims are on the path: their notes are scheduled on the audio clock, so the melody is exact
    Lv.glimSrcs = D.level.beatmap.filter(ev => ev.action === 'G' && B.glims['glim_' + ev.note]).map(ev => src(B.glims['glim_' + ev.note], E.sfx, Lv.t0 + ev.t, {gain:.85}));
    hint(auto ? 'Watching: every move lands on the music' : 'Your turn! Space jump · F punch · ↓ slide');
  }
  function stopLevel(){
    if (Lv.src) try { Lv.src.s.stop(); } catch (e) {}
    for (const x of Lv.glimSrcs || []) try { x.s.stop(); } catch (e) {}
    Lv.on = false; Lv.src = null; Lv.glimSrcs = [];
  }
  const ACT = {J:'jump', P:'punch', S:'slide', G:'glim'};
  const WIN = {perfect:.065, good:.14};
  function input(kind){
    if (!Lv.on || Lv.auto) return null;
    const t = ctx.currentTime - Lv.t0;
    Hero.act(kind);
    let best = null;
    for (const ev of D.level.beatmap){
      if (ACT[ev.action] !== kind || Lv.judged.has(ev)) continue;
      const d = t - ev.t; if (Math.abs(d) <= WIN.good && (!best || Math.abs(d) < Math.abs(best.d))) best = {ev, d};
    }
    if (!best) return null;
    Lv.judged.add(best.ev);
    const grade = Math.abs(best.d) <= WIN.perfect ? 'perfect' : 'good';
    Lv.hits[grade]++; Lv.pops.push({text:grade === 'perfect' ? 'PERFECT!' : 'GOOD', t:ctx.currentTime, col:grade === 'perfect' ? '#ffd23f' : '#9be7ff'});
    emit('judgement', {grade, action:kind, offset:best.d, bar:best.ev.bar});
    return grade;
  }
  function levelTick(){
    if (!Lv.on) return;
    const t = ctx.currentTime - Lv.t0;
    for (const ev of D.level.beatmap){
      if (ev.t > t) break;
      if (ev.done) continue;
      if (Lv.auto && ev.action !== 'G' && t >= ev.t){ ev.done = true; Hero.act(ACT[ev.action], ev); emit('action', ev); }
      else if (ev.action === 'G' && t >= ev.t){ ev.done = true; Lv.glims++; Glims.burst(false, true); emit('action', ev); }
      else if (!Lv.auto && t > ev.t + WIN.good && !Lv.judged.has(ev)){
        ev.done = true; Lv.judged.add(ev); Lv.hits.miss++; Hero.bonk();
        Lv.pops.push({text:'MISS', t:ctx.currentTime, col:'#ff6b6b'}); emit('judgement', {grade:'miss', action:ACT[ev.action], bar:ev.bar});
      } else if (!Lv.auto && Lv.judged.has(ev)) ev.done = true;
    }
    const bpb = 60 / D.level.bpm * 4, barN = Math.floor(t / bpb);
    if (barN !== Lv.lastBar && t >= 0){ Lv.lastBar = barN; emit('bar', {bar:barN, time:t}); }
    const sec = D.level.sections.filter(s => s.t <= t + 1e-3).length - 1;
    if (sec !== Lv.lastSection && sec >= 0){ Lv.lastSection = sec; emit('section', D.level.sections[sec]); }
    qs('#sSection').textContent = sec >= 0 ? D.level.sections[sec].name : '—';
    qs('#sBar').textContent = t >= 0 ? Math.min(D.level.bars, barN + 1) : '—';
    qs('#sHits').textContent = `${Lv.hits.perfect} / ${Lv.hits.good}`; qs('#sMiss').textContent = Lv.hits.miss; qs('#sGlims').textContent = Lv.glims;
  }
  function resetBeatmap(){ for (const ev of D.level.beatmap) delete ev.done; }

  // ------------------------------------------------------------------ the hero, glim bursts, scenes
  const Hero = {state:'run', t:0, ev:null,
    act(kind, ev){ this.state = kind; this.t = ctx.currentTime; this.ev = ev || null; },
    bonk(){ this.state = 'bonk'; this.t = ctx.currentTime; }};
  const Glims = {parts:[], flashCol:null, flashT:0,
    burst(king, onPath){ for (let i = 0; i < 10; i++) this.parts.push({x:onPath ? 92 : 160, y:onPath ? 118 : 70, vx:(Math.random() - .5) * 90, vy:-30 - Math.random() * 60, life:.8, col:king ? '#ffd23f' : '#ffe9a8'}); },
    flash(c){ this.flashCol = c; this.flashT = ctx.currentTime; }};
  const cv = qs('#cv'), G = cv.getContext('2d'), W_ = 320, H_ = 180;
  const THEMES = {
    briar:{sky:['#7ec8ff','#d9f0ff'], far:'#8fb7a4', near:'#4f8a4b', ground:'#6b4f2f', top:'#7fc35a', deco:'castle'},
    bayou:{sky:['#2b5a4f','#8fbf8f'], far:'#3e6b4f', near:'#2e4f36', ground:'#3b3a24', top:'#5f7f2f', deco:'mushroom'},
    fiesta:{sky:['#ff9e6b','#ffd7a1'], far:'#c2410c', near:'#7c2d12', ground:'#6b3a1f', top:'#e0a84a', deco:'flags'},
    lagoon:{sky:['#0b3a5a','#0f6b8a'], far:'#0e4f6b', near:'#0a3a4f', ground:'#2a4b5a', top:'#3fa7a0', deco:'bubbles'},
    olympus:{sky:['#5b6fd6','#e8e6ff'], far:'#c8c4e8', near:'#a5a1cf', ground:'#d9d4f0', top:'#ffffff', deco:'columns'},
    level:{sky:['#ff7ab6','#ffd23f'], far:'#7a3c8a', near:'#4a2a6a', ground:'#3a2442', top:'#c46be0', deco:'flags'},
  };
  function txt(s, x, y, size, col = '#fff', align = 'center'){
    G.font = `600 ${size}px "Pixelify Sans", monospace`; G.textAlign = align; G.textBaseline = 'middle';
    G.fillStyle = 'rgba(0,0,0,.5)'; G.fillText(s, x + 2, y + 2); G.fillStyle = col; G.fillText(s, x, y);
  }
  function scene(th, scroll, beatPh){
    const sky = G.createLinearGradient(0, 0, 0, H_); sky.addColorStop(0, th.sky[0]); sky.addColorStop(1, th.sky[1]);
    G.fillStyle = sky; G.fillRect(0, 0, W_, H_);
    // far hills
    G.fillStyle = th.far;
    for (let x = -40; x < W_ + 40; x += 4){ const wx = x + scroll * .2; G.fillRect(x, 90 + Math.sin(wx * .02) * 12 + Math.sin(wx * .053) * 6, 4, 90); }
    // decorations (parallax .5)
    const off = scroll * .5;
    for (let i = -1; i < 7; i++){
      const x = Math.floor(i * 64 - (off % 64));
      if (th.deco === 'castle'){ G.fillStyle = '#b9c7d6'; G.fillRect(x + 10, 60, 18, 50); G.fillRect(x + 8, 54, 22, 8); G.fillStyle = '#e05a5a'; G.fillRect(x + 14, 44, 10, 10); G.fillStyle = '#6a7d93'; G.fillRect(x + 17, 74, 4, 8); }
      else if (th.deco === 'mushroom'){ G.fillStyle = '#e8dcc0'; G.fillRect(x + 20, 96, 8, 26); G.fillStyle = '#c0392b'; G.fillRect(x + 8, 86, 32, 12); G.fillStyle = '#fff'; G.fillRect(x + 14, 89, 4, 3); G.fillRect(x + 28, 90, 4, 3); }
      else if (th.deco === 'flags'){ const cols = ['#ff4f79','#ffd23f','#4fd1c5','#9b5de5','#f15bb5']; for (let k = 0; k < 6; k++){ G.fillStyle = cols[(i + k + 5) % 5]; G.fillRect(x + k * 11, 30 + Math.sin(k * .9) * 2, 8, 9); } G.fillStyle = '#fff'; G.fillRect(x, 30, 64, 1); }
      else if (th.deco === 'bubbles'){ G.fillStyle = 'rgba(200,240,255,.5)'; for (let k = 0; k < 3; k++){ const y = (200 - ((ctx.currentTime * 20 + i * 37 + k * 50) % 220)); G.fillRect(x + 10 + k * 18, y, 3, 3); } G.fillStyle = '#2f8f6f'; G.fillRect(x + 40, 100, 3, 30); G.fillRect(x + 44, 108, 3, 22); }
      else if (th.deco === 'columns'){ G.fillStyle = '#f4f2ff'; G.fillRect(x + 16, 62, 12, 60); G.fillRect(x + 12, 58, 20, 6); G.fillStyle = '#d6d2f0'; G.fillRect(x + 19, 64, 2, 56); G.fillRect(x + 24, 64, 2, 56); }
    }
    // near bushes, ground (beat-bounce)
    const bounce = Math.round(Math.exp(-beatPh * 5) * 2);
    G.fillStyle = th.near;
    for (let x = -20; x < W_ + 20; x += 20){ const wx = Math.floor(x - (scroll * .8 % 20)); G.fillRect(wx, 116 - bounce - ((wx * 7) & 3), 18, 12); }
    G.fillStyle = th.ground; G.fillRect(0, 128, W_, 52);
    G.fillStyle = th.top; G.fillRect(0, 126, W_, 4);
    G.fillStyle = 'rgba(0,0,0,.15)'; for (let x = -16; x < W_ + 16; x += 16) G.fillRect(Math.floor(x - (scroll % 16)), 134, 8, 4);
  }
  function hero(x, y, t){
    const st = Hero.state, dt = ctx.currentTime - Hero.t;
    let yy = y, squash = 0, fist = 0, slide = 0;
    if (st === 'jump' && dt < .38) yy = y - Math.sin(Math.PI * dt / .38) * 30;
    else if (st === 'punch' && dt < .22) fist = 1;
    else if (st === 'slide' && dt < .4) slide = 1;
    else if (st === 'bonk' && dt < .4) squash = 1;
    const run = Math.floor(t * 10) % 2;
    G.save(); G.translate(x, yy);
    if (slide){ G.fillStyle = '#ffd23f'; G.fillRect(-10, -6, 22, 6); G.fillStyle = '#3fbf6f'; G.fillRect(-12, -10, 8, 5); }
    else {
      G.fillStyle = '#2f6fdf'; G.fillRect(-5, -10 + squash * 3, 10, 9 - squash * 3);       // body
      G.fillStyle = '#ffe0b8'; G.fillRect(-4, -18 + squash * 3, 9, 8);                     // head
      G.fillStyle = '#3fbf6f'; G.fillRect(-6, -21 + squash * 3, 12, 3); G.fillRect(3, -24 + squash * 3, 5, 3); // leaf cap
      G.fillStyle = '#111'; G.fillRect(2, -15 + squash * 3, 2, 2);
      G.fillStyle = '#ffe0b8'; G.fillRect(fist ? 7 : 5, -8, fist ? 9 : 3, 3);               // fist
      G.fillStyle = '#7a4a2a'; G.fillRect(-4 + run, -1, 3, 2); G.fillRect(1 - run, -1, 3, 2);
      if (fist){ G.fillStyle = '#fff'; G.fillRect(17, -10, 2, 7); G.fillRect(16, -7, 4, 1); }
      if (squash){ G.fillStyle = '#ffd23f'; G.fillRect(-8, -26, 2, 2); G.fillRect(6, -27, 2, 2); }
    }
    G.restore();
  }
  function obstacles(t, pps){
    const hx = 92;
    for (const ev of D.level.beatmap){
      const x = hx + (ev.t - t) * pps; if (x < -20) continue; if (x > W_ + 20) break;
      const done = ev.done || Lv.judged.has(ev);
      if (ev.action === 'J'){ G.fillStyle = done && x < hx ? '#8a5a2a' : '#b5793a'; G.fillRect(x - 6, 116, 12, 12); G.fillStyle = '#6b4220'; G.fillRect(x - 6, 116, 12, 2); G.fillRect(x - 1, 116, 2, 12); }
      else if (ev.action === 'P'){ const y = done && x < hx ? 132 : 108; G.fillStyle = '#9b5de5'; G.fillRect(x - 5, y, 10, 18); G.fillStyle = '#fff'; G.fillRect(x - 3, y + 4, 2, 2); G.fillRect(x + 1, y + 4, 2, 2); G.fillStyle = '#ffd23f'; G.fillRect(x - 5, y - 3, 10, 3); }
      else if (ev.action === 'S'){ G.fillStyle = '#5a3a1a'; G.fillRect(x - 10, 102, 30, 6); G.fillStyle = '#3f8f3f'; G.fillRect(x - 10, 98, 30, 4); }
      else if (ev.action === 'G' && !ev.done){ const b = Math.sin(ctx.currentTime * 8 + ev.t * 3) * 2; G.fillStyle = '#fff6c8'; G.fillRect(x - 2, 96 + b, 5, 5); G.fillStyle = '#ffd23f'; G.fillRect(x - 1, 97 + b, 3, 3); }
    }
  }
  function draw(){
    requestAnimationFrame(draw);
    if (!ctx || !D) return;
    const now = ctx.currentTime;
    if (Lv.on || mode === 'level'){
      levelTick();
      const t = Lv.on ? now - Lv.t0 : 0, beat = 60 / D.level.bpm, pps = 64 / beat;
      scene(THEMES.level, t * pps, ((t % beat) + beat) % beat / beat);
      obstacles(t, pps);
      hero(92, 128, now);
      const sec = Lv.lastSection >= 0 ? D.level.sections[Lv.lastSection].name : 'Kazoo Rumble';
      txt(sec, 8, 12, 10, '#fff', 'left');
      for (const p of Lv.pops.slice(-3)){ const a = now - p.t; if (a < .7) txt(p.text, 92, 76 - a * 30, 12, p.col); }
      if (!Lv.on){ txt('KAZOO RUMBLE', W_ / 2, 52, 22, '#ffd23f'); txt('a musical level', W_ / 2, 74, 11); }
    } else {
      const w = Wd.world || D.worlds[0], th = THEMES[w.id], b = bar(w), beat = 60 / w.bpm;
      const t = Wd.playing ? now - Wd.t0 : 0, pps = 40 / beat;
      scene(th, t * pps, Wd.playing ? ((t % beat) + beat) % beat / beat : 0);
      // a row of floating glims ahead of the hero
      const kingOn = Wd.king && now < Wd.kingUntil;
      for (let i = 0; i < 6; i++){ const x = 150 + i * 22 - ((t * pps) % 22), y = 70 + Math.sin(now * 3 + i) * 4;
        G.fillStyle = kingOn ? '#ffd23f' : '#fff6c8'; G.fillRect(x, y, 5, 5); G.fillStyle = kingOn ? '#ff9f1c' : '#ffd23f'; G.fillRect(x + 1, y + 1, 3, 3); }
      hero(110, 128, now);
      txt(w.name, 8, 12, 11, '#fff', 'left');
      txt(`${w.key} · ${w.bpm} BPM · ${['', 'explore', 'action', 'chase'][Wd.layer]}`, 8, 26, 8, 'rgba(255,255,255,.85)', 'left');
      if (Wd.playing && now >= Wd.t0){ const bi = barIndex(w); txt(`bar ${bi + 1}/${w.bars} · ${w.chords[bi]}`, W_ - 8, 12, 9, '#fff', 'right'); if (Wd.lastBar !== bi){ Wd.lastBar = bi; emit('bar', {bar:bi, chord:w.chords[bi], world:w.id}); } }
      if (Wd.auto && Wd.playing){ const cyc = Math.floor((now - Wd.t0) / (b * 4)) % 4, want = [1, 2, 3, 2][cyc]; if (want !== Wd.layer) setLayer(want); }
    }
    if (Wd.banner && now - Wd.banner.t < 1.6) txt(Wd.banner.text, W_ / 2, 50, 18, '#ffd23f');
    for (let i = Glims.parts.length - 1; i >= 0; i--){ const p = Glims.parts[i]; p.x += p.vx / 60; p.y += p.vy / 60; p.vy += 4; p.life -= 1 / 60;
      if (p.life <= 0){ Glims.parts.splice(i, 1); continue; } G.globalAlpha = Math.min(1, p.life * 2); G.fillStyle = p.col; G.fillRect(p.x | 0, p.y | 0, 2, 2); }
    G.globalAlpha = 1;
    if (Glims.flashCol && now - Glims.flashT < .25){ G.globalAlpha = .35 * (1 - (now - Glims.flashT) / .25); G.fillStyle = Glims.flashCol; G.fillRect(0, 0, W_, H_); G.globalAlpha = 1; }
  }

  // ------------------------------------------------------------------ UI
  let mode = 'worlds';
  function setMode(m){
    mode = m; qs('#tabWorlds').setAttribute('aria-selected', m === 'worlds'); qs('#tabLevel').setAttribute('aria-selected', m === 'level');
    qs('#worldsPanel').hidden = m !== 'worlds'; qs('#levelPanel').hidden = m !== 'level';
    if (m === 'level'){ stopWorld(); hint('Kazoo Rumble — watch it, or play it yourself'); loadLevel().catch(e => console.warn(e)); }
    else { stopLevel(); hint('Pick a world and press Play'); }
  }
  qs('#tabWorlds').addEventListener('click', () => setMode('worlds'));
  qs('#tabLevel').addEventListener('click', () => setMode('level'));
  qs('#worldPlay').addEventListener('click', async e => { e.currentTarget.blur(); await ctx.resume(); Wd.playing ? stopWorld() : playWorld(Wd.world || D.worlds[0]); });
  qsa('[data-tier]').forEach(b => b.addEventListener('click', () => { Wd.auto = false; qs('#autoLayer').setAttribute('aria-pressed', false); setLayer(+b.dataset.tier); }));
  qs('#autoLayer').addEventListener('click', e => { Wd.auto = !Wd.auto; e.currentTarget.setAttribute('aria-pressed', Wd.auto); });
  qs('#collect').addEventListener('click', () => collect());
  qs('#kingBtn').addEventListener('click', () => { const on = !(Wd.king && ctx.currentTime < Wd.kingUntil); king(on); });
  qsa('[data-sting]').forEach(b => b.addEventListener('click', async () => { await ctx.resume(); sting(b.dataset.sting); }));
  qs('#levelAuto').addEventListener('click', async e => { e.currentTarget.blur(); await ctx.resume(); resetBeatmap(); startLevel({auto:true}); });
  qs('#levelPlay').addEventListener('click', async e => { e.currentTarget.blur(); await ctx.resume(); resetBeatmap(); startLevel({auto:false}); });
  qs('#levelStop').addEventListener('click', () => stopLevel());
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT') return;
    const k = e.code;
    if (mode === 'worlds' && k === 'KeyG'){ e.preventDefault(); collect(); return; }
    if (mode !== 'level') return;
    const kind = (k === 'Space' || k === 'ArrowUp') ? 'jump' : (k === 'KeyF' || k === 'ArrowRight') ? 'punch' : (k === 'ArrowDown' || k === 'KeyS') ? 'slide' : null;
    if (kind){ e.preventDefault(); if (e.target.tagName === 'BUTTON') e.target.blur(); input(kind); }
  });
  qs('#screen').addEventListener('pointerdown', e => { e.preventDefault(); if (mode === 'worlds') collect(); else input('jump'); });

  function buildUI(){
    const wc = qs('#worldChips');
    for (const w of D.worlds){
      const b = document.createElement('button'); b.className = 'chip'; b.type = 'button'; b.dataset.world = w.id;
      b.innerHTML = `${w.name} <span class="status">${w.meter} · ${w.bpm}</span>`; b.title = w.mood;
      b.addEventListener('click', async () => { await ctx.resume(); switchWorld(w); });
      wc.appendChild(b);
    }
    qsa('[data-world]')[0].setAttribute('aria-pressed', true);
    // beatmap lanes
    const lanes = qs('#lanes'), L = D.level, end = L.music_end;
    for (const [a, name, col] of [['J', 'jump', '#ffd23f'], ['P', 'punch', '#9b5de5'], ['S', 'slide', '#4fd1c5'], ['G', 'glim', '#fff6c8']]){
      const lab = document.createElement('div'); lab.textContent = name; lanes.appendChild(lab);
      const c = document.createElement('canvas'); c.width = 900; c.height = 14; const g = c.getContext('2d');
      g.fillStyle = 'rgba(255,255,255,.08)'; for (const s of L.sections) g.fillRect(s.t / end * 900, 0, 1, 14);
      g.fillStyle = col; for (const ev of L.beatmap) if (ev.action === a) g.fillRect(ev.t / end * 900, 2, 2, 10);
      lanes.appendChild(c);
    }
  }

  window.CanopyGame = Object.freeze({
    mode:'legends',
    setWorld:id => { const w = D.worlds.find(x => x.id === id); if (!w) throw new TypeError('Unknown world ' + id); return switchWorld(w); },
    play:() => playWorld(Wd.world || D.worlds[0]), stop:() => { stopWorld(); stopLevel(); },
    setLayer, collect, king, sting, startLevel, stopLevel, input,
    getState:() => ({mode, world:Wd.world && Wd.world.id, playing:Wd.playing, layer:Wd.layer,
      bar:Wd.playing ? barIndex(Wd.world) : null, chord:Wd.playing ? Wd.world.chords[barIndex(Wd.world)] : null,
      king:Wd.king && ctx && ctx.currentTime < Wd.kingUntil,
      level:{on:Lv.on, auto:Lv.auto, time:Lv.on ? ctx.currentTime - Lv.t0 : null, hits:{...Lv.hits}, glims:Lv.glims,
        section:Lv.lastSection >= 0 ? D.level.sections[Lv.lastSection].name : null}}),
    on:(n, fn) => { if (!listeners[n]) throw new TypeError('Unknown event ' + n); listeners[n].add(fn); return () => listeners[n].delete(fn); },
    off:(n, fn) => listeners[n] && listeners[n].delete(fn),
  });

  (async () => {
    D = await (await fetch('legends/legends.json')).json();
    ctx = new AudioContext({latencyHint:'interactive'});
    E.master = ctx.createGain(); E.master.connect(ctx.destination);
    E.lp = ctx.createBiquadFilter(); E.lp.type = 'lowpass'; E.lp.frequency.value = 20000; E.lp.connect(E.master);
    E.music = ctx.createGain(); E.music.connect(E.lp);
    E.sfx = ctx.createGain(); E.sfx.gain.value = .9; E.sfx.connect(E.master);
    Wd.world = D.worlds[0];
    buildUI();
    hint('Loading Briar Keep…');
    await loadWorld(D.worlds[0]); await loadGlims();
    hint('Pick a world and press Play · G collects a glim');
    for (const w of D.worlds.slice(1)) loadWorld(w).catch(e => console.warn(e));
  })().catch(e => { console.warn(e); hint('Could not load: ' + e.message + ' (serve the player folder over http)'); });
  requestAnimationFrame(draw);
})();
