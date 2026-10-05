/* Microgame Rush: a WarioWare-style run sequenced from bar-exact cues (src/tracks/microgame.py).
   Data: microgame/microgame.json. Audio: one sprite per tempo tier (microgame/tierN.m4a, or the
   same bytes as base64 in tierN.b64.txt). Every cue starts on the audio clock exactly where the
   previous one ends; loops ring out through their rendered tails. */
'use strict';
(() => {
  const qs = s => document.querySelector(s);
  const SOUNDTRACKS = [['jungle','Jungle'],['space','Space'],['canopy','Jungle Canopy Level'],['mine','Mine Cart Level'],
    ['underwater','Underwater Level'],['orbital','Orbital Rainforest'],['crystal','Crystal Caverns'],
    ['greenhouse','Clockwork Greenhouse'],['kapok','KAPOK (Oldfield-style suite)'],['microgame','Microgame Rush (WarioWare-style)']];
  const sel = qs('#soundscape');
  for (const [v, n] of SOUNDTRACKS){ const o = document.createElement('option'); o.value = v; o.textContent = n; sel.appendChild(o); }
  sel.value = 'microgame';
  sel.addEventListener('change', () => { location.href = sel.value === 'kapok' ? 'kapok.html' : 'index.html?soundscape=' + sel.value; });

  const TXT = {
    en:{tap:'TAP!', tap3:'TAP ×3!', wait:"DON'T TAP!", green:'WAIT FOR GREEN!', stop:'STOP ON THE STAR!', mash:'MASH!',
        win:'NICE!', lose:'OOPS!', speedup:'SPEED UP!', boss:'BOSS STAGE!', over:'GAME OVER', clear:'CLEAR!', ready:'GET READY…', game:'GAME',
        hint:'Click, tap or press Space to play'},
    es:{tap:'¡TOCA!', tap3:'¡TOCA 3 VECES!', wait:'¡NO TOQUES!', green:'¡ESPERA EL VERDE!', stop:'¡PARA EN LA ESTRELLA!', mash:'¡RÁPIDO, RÁPIDO!',
        win:'¡BIEN!', lose:'¡UY!', speedup:'¡MÁS RÁPIDO!', boss:'¡JEFE FINAL!', over:'FIN DEL JUEGO', clear:'¡SUPERADO!', ready:'¡PREPÁRATE!', game:'JUEGO',
        hint:'Haz clic, toca o pulsa Espacio para jugar'},
  };
  const MECHS = ['tap', 'tap3', 'wait', 'green', 'stop'];
  const PAL = {chip:['#1a1c4a','#3b46c4','#7ef0ff'], funk:['#3a0f2a','#b8336a','#ffd166'], mariachi:['#3d1a05','#c2410c','#facc15'],
    surf:['#06324a','#0ea5b5','#fef08a'], jungle:['#0f2e1a','#2f8f4a','#c6f68d'], space:['#0b0420','#6d28d9','#f0abfc'],
    toybox:['#3b1d3b','#e879a7','#fff1a8'], polka:['#3a1717','#dc2626','#fde68a'], boss:['#1a0505','#7f1d1d','#ff4f4f'],
    stage:['#17101f','#5b2a86','#ffc93c']};
  let W = null, ctx = null, T = [], lang = 'en', auto = false, viewTier = 0;
  const E = {};
  const listeners = {cue:new Set(), beat:new Set(), microgame:new Set(), result:new Set(), speedup:new Set(), boss:new Set(), gameover:new Set()};
  const emit = (n, x) => listeners[n].forEach(fn => { try { fn(x); } catch (e) { console.warn(e); } });
  const hint = t => qs('#hint').textContent = t;
  const tx = k => TXT[lang][k];

  // ------------------------------------------------------------------ loading (sync impulse makes codec padding harmless)
  async function bytes(file, b64){
    try { const r = await fetch(file); if (r.ok){ const b = await r.arrayBuffer(); if (b.byteLength > 1000 && !/^<|^version https/.test(new TextDecoder().decode(b.slice(0, 40)))) return b; } } catch (e) {}
    const t = await (await fetch(b64)).text(), bin = atob(t), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }
  function slice(buf, start, len){
    const out = ctx.createBuffer(2, Math.max(1, len), ctx.sampleRate);
    for (let c = 0; c < 2; c++) out.copyToChannel(buf.getChannelData(Math.min(c, buf.numberOfChannels - 1)).subarray(start, start + len), c);
    return out;
  }
  async function loadTier(i){
    const tier = W.tiers[i], buf = await ctx.decodeAudioData(await bytes(tier.file, tier.b64));
    const k = ctx.sampleRate / W.sr, d = buf.getChannelData(0);
    let p = 0, mx = 0; for (let j = 0; j < Math.round(7000 * k); j++){ const v = Math.abs(d[j]); if (v > mx){ mx = v; p = j; } }
    const base = p - Math.round(tier.sync * k), cues = {};
    for (const [name, c] of Object.entries(tier.cues)){
      cues[name] = {kind:c.kind, dur:c.length / W.sr, buf:slice(buf, base + Math.round(c.start * k), Math.round(c.total * k)),
        tail:c.tail ? slice(buf, base + Math.round(c.tail * k), Math.round(c.tail_len * k)) : null};
    }
    T[i] = {bpm:tier.bpm, bar:240 / tier.bpm, beat:60 / tier.bpm, cues};
  }
  async function load(){
    W = await (await fetch('microgame/microgame.json')).json();
    ctx = new AudioContext({latencyHint:'interactive'});
    E.master = ctx.createGain(); E.master.connect(ctx.destination);
    E.music = ctx.createGain(); E.music.connect(E.master);
    E.sfx = ctx.createGain(); E.sfx.gain.value = .8; E.sfx.connect(E.master);
    hint('Loading speed 1…');
    await loadTier(0);
    hint(tx('hint') + ' · press Start run');
    buildBoard();
    for (let i = 1; i < W.tiers.length; i++) await loadTier(i).catch(e => console.warn(e));
    buildBoard();
  }

  // ------------------------------------------------------------------ playback
  const live = new Set();
  function play(tierI, name, t, {passes = 1, tail = true} = {}){
    const c = T[tierI].cues[name], src = ctx.createBufferSource();
    src.buffer = c.buf; src.connect(E.music); live.add(src); src.onended = () => live.delete(src);
    const dur = c.dur * passes;
    if (c.kind === 'loop'){
      src.loop = true; src.loopEnd = c.dur; src.start(t); src.stop(t + dur);
      if (tail && c.tail){ const s2 = ctx.createBufferSource(); s2.buffer = c.tail; s2.connect(E.music); s2.start(t + dur); live.add(s2); s2.onended = () => live.delete(s2); }
    } else src.start(t);
    log(t, name, tierI);
    emit('cue', {name, tier:tierI, bpm:T[tierI].bpm, time:t, duration:dur});
    return t + dur;
  }
  function stopAll(){ for (const s of live) try { s.stop(); } catch (e) {} live.clear(); }
  function sfx(name, opts = {}){
    const tier = T[R.tier] || T[0];
    const r = CanopyEvents.schedule(name, {bpm:tier.bpm, now:ctx.currentTime, origin:R.step ? R.step.t0 : ctx.currentTime, chord:'C',
      key:'C major', chain:R.chain, palette:{lead:name === 'collect' ? 'glass' : 'kalimba'}, opts});
    R.chain = r.chain; CanopyEvents.play(ctx, E.sfx, r.notes, 1);
    for (const f of r.fx) if (f.type === 'duck'){ E.music.gain.setTargetAtTime(.45, f.time, .02); E.music.gain.setTargetAtTime(1, f.time + .3, .2); }
  }

  // ------------------------------------------------------------------ the run
  const R = {running:false, tier:0, round:0, lives:4, score:0, steps:[], step:null, nextT:0, plan:null, chain:null, lastStyle:null};
  function startRun(){
    if (!T[0]) return;
    ctx.resume(); stopAll();
    Object.assign(R, {running:true, tier:0, round:0, lives:4, score:0, steps:[], step:null, chain:null, plan:'interlude', nextT:ctx.currentTime + .15});
    qs('#start').textContent = 'Stop run'; hint(tx('hint'));
  }
  function stopRun(){ R.running = false; stopAll(); qs('#start').textContent = 'Start run'; }
  function push(step){ R.steps.push(step); if (R.steps.length > 12) R.steps.shift(); return step; }
  function tierReady(i){ return !!T[i]; }

  function planNext(){
    const t = R.nextT, ti = R.tier, tier = T[ti];
    switch (R.plan){
      case 'interlude': {
        const end = play(ti, 'interlude', t);
        push({type:'interlude', t0:t, t1:end, tier:ti});
        R.nextT = end; R.plan = (R.round + 1) % 12 === 0 ? 'boss_intro' : 'microgame';
        break;
      }
      case 'microgame': {
        let style; do style = W.styles[Math.floor(Math.random() * W.styles.length)]; while (style === R.lastStyle);
        R.lastStyle = style; R.round++;
        const mech = MECHS[Math.floor(Math.random() * MECHS.length)];
        const end = play(ti, 'mg_' + style, t);
        const mg = {type:'microgame', style, mech, t0:t, t1:end, tier:ti, presses:0, outcome:null,
          greenAt:t + tier.beat * (2.5 + Math.floor(Math.random() * 4)), wasPressedAt:null};
        R.step = push(mg); R.nextT = end; R.plan = 'result';
        emit('microgame', {round:R.round, style, mechanic:mech, start:t, end, bpm:tier.bpm, beats:8});
        if (auto) bot(mg);
        break;
      }
      case 'boss_intro': {
        const end = play(ti, 'boss_intro', t);
        push({type:'banner', key:'boss', t0:t, t1:end, tier:ti, style:'boss'});
        R.nextT = end; R.plan = 'boss'; R.round++;
        emit('boss', {round:R.round, time:t});
        break;
      }
      case 'boss': {
        const end = play(ti, 'boss', t, {passes:2});
        const mg = {type:'microgame', style:'boss', mech:'mash', t0:t, t1:end, tier:ti, presses:0, need:20, outcome:null};
        R.step = push(mg); R.nextT = end; R.plan = 'result';
        emit('microgame', {round:R.round, style:'boss', mechanic:'mash', start:t, end, bpm:tier.bpm, beats:32});
        if (auto) bot(mg);
        break;
      }
      case 'result': {
        const mg = R.step, isBoss = mg.style === 'boss';
        const win = mg.outcome ? mg.outcome === 'win' : (mg.mech === 'wait');
        mg.outcome = win ? 'win' : 'lose';
        if (win) R.score++; else R.lives--;
        const cue = isBoss ? (win ? 'boss_win' : 'boss_lose') : (win ? 'win' : 'lose');
        play(ti, cue, t);
        const end = t + T[ti].cues[cue].dur;
        push({type:'result', win, t0:t, t1:end, tier:ti, style:mg.style});
        emit('result', {win, round:R.round, score:R.score, lives:R.lives, time:t});
        R.nextT = end;
        if (R.lives <= 0) R.plan = 'gameover';
        else if (isBoss && win) R.plan = 'clear';
        else if (!isBoss && R.round % 4 === 0 && R.tier < 3 && tierReady(R.tier + 1)) R.plan = 'speedup';
        else R.plan = 'interlude';
        break;
      }
      case 'speedup': case 'clear': {
        const name = R.plan, end = play(ti, name, t);
        push({type:'banner', key:name, t0:t, t1:end, tier:ti, style:'stage'});
        R.nextT = end; R.plan = 'interlude';
        if (R.tier < 3 && tierReady(R.tier + 1)){ R.tier++; emit('speedup', {tier:R.tier, bpm:T[R.tier].bpm, time:end}); }
        break;
      }
      case 'gameover': {
        const end = play(ti, 'gameover', t);
        push({type:'banner', key:'over', t0:t, t1:end + 3, tier:ti, style:'stage'});
        emit('gameover', {score:R.score, round:R.round, time:t});
        R.plan = null; setTimeout(() => { R.running = false; qs('#start').textContent = 'Start run'; }, (end - ctx.currentTime) * 1000 + 300);
        break;
      }
    }
  }
  // the result is decided a hair before the bar line, so the jingle lands exactly on it
  function tick(){
    if (R.running && R.plan){
      const look = R.plan === 'result' ? .09 : .2;
      if (ctx.currentTime >= R.nextT - look) planNext();
    }
    const s = R.steps.find(x => x.t0 <= ctx.currentTime && ctx.currentTime < x.t1);
    if (s && s.type === 'microgame') R.step = s;
  }
  setInterval(() => { if (ctx) tick(); }, 15);

  // ------------------------------------------------------------------ input and the five microgames
  function stopPos(mg, t){ const b = T[mg.tier].beat; return .5 + .44 * Math.sin(2 * Math.PI * (t - mg.t0) / (b * 2.6)); }
  function press(){
    if (!ctx) return;
    ctx.resume();
    if (!R.running){ startRun(); return; }
    const t = ctx.currentTime, mg = R.steps.find(x => x.type === 'microgame' && x.t0 <= t && t < x.t1);
    if (!mg || mg.outcome) return;
    mg.presses++;
    sfx('jump', {sound:true});
    let res = null;
    if (mg.mech === 'tap') res = 'win';
    else if (mg.mech === 'tap3') res = mg.presses >= 3 ? 'win' : null;
    else if (mg.mech === 'wait') res = 'lose';
    else if (mg.mech === 'green') res = t >= mg.greenAt ? 'win' : 'lose';
    else if (mg.mech === 'stop') res = Math.abs(stopPos(mg, t) - .5) < .085 ? 'win' : 'lose';
    else if (mg.mech === 'mash') res = mg.presses >= mg.need ? 'win' : null;
    if (res) resolve(res === 'win', mg);
  }
  function resolve(win, mg = R.steps.find(x => x.type === 'microgame' && !x.outcome)){
    if (!mg || mg.outcome) return false;            // resolve() fires once per microgame
    mg.outcome = win ? 'win' : 'lose'; mg.resolvedAt = ctx.currentTime;
    sfx(win ? 'collect' : 'hurt');
    return true;
  }
  function bot(mg){
    const good = Math.random() < .82, b = T[mg.tier].beat, at = (dt, n = 1) => setTimeout(() => { for (let i = 0; i < n; i++) press(); }, Math.max(0, (mg.t0 - ctx.currentTime + dt) * 1000));
    if (mg.mech === 'tap') at(b * (1 + Math.random() * 4));
    else if (mg.mech === 'tap3') { if (good) [1.2, 2.4, 3.6].forEach(x => at(b * x)); else [1.5, 3].forEach(x => at(b * x)); }
    else if (mg.mech === 'wait') { if (!good) at(b * 4); }
    else if (mg.mech === 'green') at(good ? mg.greenAt - mg.t0 + .18 : mg.greenAt - mg.t0 - b * .8);
    else if (mg.mech === 'stop') {
      const P = b * 2.6, k = Math.floor(2 * (b * 2) / P) + 1, hitT = good ? (k * P) / 2 : (k * P) / 2 + P / 4;
      at(hitT);
    } else if (mg.mech === 'mash') { const n = good ? 22 : 12; for (let i = 0; i < n; i++) at(b * (1 + i * .9)); }
  }
  qs('#screen').addEventListener('pointerdown', e => { e.preventDefault(); press(); });
  document.addEventListener('keydown', e => { if (e.code === 'Space' || e.code === 'Enter'){ if (e.target.tagName === 'BUTTON' || e.target.tagName === 'SELECT'){ if (e.code === 'Enter') return; e.target.blur(); } e.preventDefault(); press(); } });
  qs('#start').addEventListener('click', e => { e.currentTarget.blur(); if (!ctx) return; ctx.resume(); R.running ? stopRun() : startRun(); });
  qs('#lang').addEventListener('click', e => { lang = lang === 'en' ? 'es' : 'en'; e.currentTarget.setAttribute('aria-pressed', lang === 'es'); e.currentTarget.textContent = lang === 'es' ? 'English' : 'Español'; hint(tx('hint')); });
  qs('#auto').addEventListener('click', e => { auto = !auto; e.currentTarget.setAttribute('aria-pressed', auto); });

  // ------------------------------------------------------------------ cue board + log
  function buildBoard(){
    const tp = qs('#tierPick'); tp.innerHTML = '';
    W.tiers.forEach((t, i) => { const b = document.createElement('button'); b.className = 'chip'; b.type = 'button';
      b.textContent = `Speed ${i + 1} · ${t.bpm}`; b.disabled = !T[i]; b.setAttribute('aria-pressed', i === viewTier);
      b.addEventListener('click', () => { viewTier = i; buildBoard(); }); tp.appendChild(b); });
    const box = qs('#cues'); box.innerHTML = '';
    if (!T[viewTier]) return;
    for (const [name, c] of Object.entries(T[viewTier].cues)){
      const b = document.createElement('button'); b.type = 'button';
      const label = name.startsWith('mg_') ? W.style_names[name.slice(3)] : name.replace(/_/g, ' ');
      b.innerHTML = `${label}<small>${c.kind} · ${(c.dur).toFixed(2)} s</small>`;
      b.addEventListener('click', () => { ctx.resume(); if (R.running) stopRun(); stopAll(); play(viewTier, name, ctx.currentTime + .05); });
      box.appendChild(b);
    }
  }
  function log(t, name, ti){
    const el = document.createElement('div'), lab = name.startsWith('mg_') ? W.style_names[name.slice(3)] : name;
    el.innerHTML = `${t.toFixed(3)} s · <b>${lab}</b> · ${T[ti].bpm} BPM`;
    const L = qs('#log'); L.prepend(el); while (L.children.length > 60) L.lastChild.remove();
  }

  // ------------------------------------------------------------------ drawing (320×180, chunky pixels)
  const cv = qs('#cv'), G = cv.getContext('2d'), WD = 320, HT = 180;
  function txt(s, x, y, size, col = '#fff', align = 'center'){
    G.font = `600 ${size}px "Pixelify Sans", monospace`; G.textAlign = align; G.textBaseline = 'middle';
    G.fillStyle = 'rgba(0,0,0,.45)'; G.fillText(s, x + 2, y + 2); G.fillStyle = col; G.fillText(s, x, y);
  }
  function heart(x, y, on){ G.fillStyle = on ? '#ff4f79' : '#4a3a5a';
    [[1,0,2],[4,0,2],[0,1,7],[0,2,7],[1,3,5],[2,4,3],[3,5,1]].forEach(([dx, dy, w]) => G.fillRect(x + dx * 2, y + dy * 2, w * 2, 2)); }
  function bg(style, t, beat){
    const [a, b2] = PAL[style] || PAL.stage, ph = (t / beat) % 1;
    G.fillStyle = a; G.fillRect(0, 0, WD, HT);
    G.fillStyle = b2; const s = 20, off = (t * 12) % (s * 2);
    for (let y = -s * 2; y < HT + s; y += s) for (let x = -s * 2; x < WD + s; x += s * 2) G.fillRect(x + ((y / s) % 2 ? s : 0) + off, y + off * .5, s * .5, s * .5);
    return Math.exp(-ph * 6);
  }
  function draw(){
    requestAnimationFrame(draw);
    if (!ctx){ return; }
    const t = ctx.currentTime, tierI = R.running ? R.tier : viewTier, tier = T[tierI] || T[0];
    qs('#sScore').textContent = R.score; qs('#sLives').textContent = R.lives; qs('#sTier').textContent = R.tier + 1;
    qs('#sBpm').textContent = tier ? tier.bpm : '—';
    if (!tier){ G.fillStyle = '#000'; G.fillRect(0, 0, WD, HT); txt('Loading…', WD / 2, HT / 2, 16); return; }
    const s = R.steps.find(x => x.t0 <= t && t < x.t1) || R.steps[R.steps.length - 1];
    if (!s){
      bg('stage', t, tier.beat); txt('MICROGAME', WD / 2, 62, 30, '#ffc93c'); txt('RUSH', WD / 2, 92, 30);
      txt(lang === 'es' ? 'Pulsa para empezar' : 'Press to start', WD / 2, 128, 12, '#e9d5ff'); return;
    }
    const st = T[s.tier], beat = st.beat, pulse = bg(s.style || 'stage', t, beat), bob = Math.round(pulse * 3);
    const el = t - s.t0;
    if (s.type === 'interlude'){
      txt(`${tx('game')} ${R.round + 1}`, WD / 2, 34, 16, '#e9d5ff');
      txt(String(R.score), WD / 2, 78 - bob, 44, '#ffc93c');
      for (let i = 0; i < 4; i++) heart(WD / 2 - 46 + i * 24, 116, i < R.lives);
      if (el > st.bar) txt(tx('ready'), WD / 2, 150, 14);
      qs('#sCue').textContent = 'Interlude';
    } else if (s.type === 'microgame'){
      const pal = PAL[s.style], frac = Math.min(1, el / (s.t1 - s.t0)), beatsLeft = (s.t1 - t) / beat;
      txt(tx(s.mech), WD / 2, el < beat * 1.5 ? 30 - bob : 22, el < beat * 1.5 ? 22 : 13, pal[2]);
      txt(W.style_names[s.style] || '', 6, 172, 8, 'rgba(255,255,255,.6)', 'left');
      const cx = WD / 2, cy = 92;
      if (s.mech === 'tap' || s.mech === 'tap3'){
        const n = s.mech === 'tap' ? 1 : 3;
        for (let i = 0; i < n; i++){ const x = cx + (i - (n - 1) / 2) * 52, on = s.presses > i || s.outcome === 'win';
          G.fillStyle = on ? '#4ade80' : pal[2]; G.beginPath(); G.arc(x, cy, 18 + (on ? 0 : bob), 0, 7); G.fill();
          G.fillStyle = 'rgba(0,0,0,.35)'; G.beginPath(); G.arc(x, cy + 3, 12, 0, 7); G.fill(); }
      } else if (s.mech === 'wait'){
        G.fillStyle = s.outcome === 'lose' ? '#ff4f4f' : '#b91c1c'; G.fillRect(cx - 26, cy - 6, 52, 22); G.fillStyle = '#ef4444'; G.fillRect(cx - 20, cy - 14 + (s.outcome === 'lose' ? 6 : 0), 40, 12);
        txt('!', cx, cy - 34 - bob, 18, '#fff');
      } else if (s.mech === 'green'){
        const green = t >= s.greenAt; G.fillStyle = '#1f1f1f'; G.fillRect(cx - 16, cy - 36, 32, 72);
        G.fillStyle = green ? '#3a3a3a' : '#ef4444'; G.beginPath(); G.arc(cx, cy - 18, 11, 0, 7); G.fill();
        G.fillStyle = green ? '#4ade80' : '#3a3a3a'; G.beginPath(); G.arc(cx, cy + 18, 11, 0, 7); G.fill();
      } else if (s.mech === 'stop'){
        G.fillStyle = 'rgba(0,0,0,.35)'; G.fillRect(40, cy - 6, 240, 12);
        txt('★', cx, cy - 18, 18, '#ffd700');
        const x = 40 + 240 * (s.outcome ? stopPos(s, s.resolvedAt || t) : stopPos(s, t));
        G.fillStyle = '#fff'; G.fillRect(Math.round(x) - 2, cy - 12, 4, 24);
      } else if (s.mech === 'mash'){
        G.fillStyle = '#7f1d1d'; G.fillRect(cx - 34, cy - 30 + bob, 68, 56); G.fillStyle = '#fca5a5'; G.fillRect(cx - 22, cy - 16 + bob, 12, 10); G.fillRect(cx + 10, cy - 16 + bob, 12, 10);
        G.fillStyle = '#000'; G.fillRect(cx - 18, cy + 8 + bob, 36, 6);
        const hp = Math.max(0, 1 - s.presses / s.need); G.fillStyle = '#333'; G.fillRect(cx - 50, cy + 34, 100, 6); G.fillStyle = '#ff4f4f'; G.fillRect(cx - 50, cy + 34, 100 * hp, 6);
      }
      if (s.outcome){ txt(s.outcome === 'win' ? '✓' : '✗', WD - 34, 92, 30, s.outcome === 'win' ? '#4ade80' : '#ff4f4f'); }
      // the bomb fuse
      G.fillStyle = '#111'; G.beginPath(); G.arc(18, 152, 9, 0, 7); G.fill(); G.fillStyle = '#555'; G.fillRect(22, 141, 4, 4);
      const fx2 = 28 + (WD - 48) * (1 - frac); G.fillStyle = '#d6c08a'; G.fillRect(28, 150, Math.max(0, fx2 - 28), 3);
      G.fillStyle = (Math.floor(t * 20) % 2) ? '#ffef5a' : '#ff8a3d'; G.fillRect(fx2 - 2, 148, 5, 6);
      if (beatsLeft < 3 && !s.outcome) txt(String(Math.ceil(beatsLeft)), 18, 126 - bob, 20, '#fff');
      qs('#sCue').textContent = W.style_names[s.style];
    } else if (s.type === 'result'){
      txt(s.win ? tx('win') : tx('lose'), WD / 2, 80 - bob * 2, 40, s.win ? '#4ade80' : '#ff4f79');
      for (let i = 0; i < 4; i++) heart(WD / 2 - 46 + i * 24, 124, i < R.lives);
      qs('#sCue').textContent = s.win ? 'Win' : 'Lose';
    } else if (s.type === 'banner'){
      const col = s.key === 'boss' ? (Math.floor(el / (beat / 2)) % 2 ? '#ff4f4f' : '#fff') : '#ffc93c';
      txt(tx(s.key), WD / 2, 84 - bob * 2, s.key === 'over' ? 30 : 34, col);
      if (s.key === 'speedup') for (let i = 0; i < 6; i++){ const x = ((el * 200 + i * 60) % (WD + 40)) - 20; txt('»', x, 132, 20, '#fff'); }
      if (s.key === 'over') txt(`${R.score}`, WD / 2, 126, 22, '#e9d5ff');
      qs('#sCue').textContent = {boss:'Boss intro', speedup:'Speed up', clear:'Clear', over:'Game over'}[s.key];
    }
    if (s.t0 <= t) { const b = Math.floor(el / beat); if (b !== s._lb){ s._lb = b; emit('beat', {beat:b, step:s.type, time:s.t0 + b * beat}); } }
  }
  requestAnimationFrame(draw);

  window.CanopyGame = Object.freeze({
    mode:'microgame', start:startRun, stop:stopRun, press, resolve:win => resolve(!!win),
    getState:() => ({running:R.running, tier:R.tier, bpm:T[R.tier] ? T[R.tier].bpm : null, round:R.round, score:R.score, lives:R.lives,
      phase:(R.steps.find(x => ctx && x.t0 <= ctx.currentTime && ctx.currentTime < x.t1) || {}).type || null,
      microgame:R.step ? {style:R.step.style, mechanic:R.step.mech, outcome:R.step.outcome, start:R.step.t0, end:R.step.t1} : null, lang}),
    on:(n, fn) => { if (!listeners[n]) throw new TypeError('Unknown event ' + n); listeners[n].add(fn); return () => listeners[n].delete(fn); },
    off:(n, fn) => listeners[n] && listeners[n].delete(fn),
    setLanguage:l => { if (TXT[l]) lang = l; },
    audition:(name, tier = 0) => T[tier] && play(tier, name, ctx.currentTime + .05),
  });
  load().catch(e => { console.warn(e); hint('Could not load: ' + e.message + ' (serve the player folder over http)'); });
})();
