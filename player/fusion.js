/* Original adaptive scores: shared harmonic anchors, sample-exact 16-bar loops.
   No recordings or quoted melodies. This module also runs in a render worker. */
'use strict';
globalThis.CanopyFusion = (() => {
  const RATE = 28000; // Integer samples/beat at every configured tempo.
  const clamp = x => Math.max(0, Math.min(1, x));
  const layers = ['amb','water','wildlife','birds','woods','shaker','drums','bass','keys','pad','marimba','kalimba'];
  const modes = {
    canopy: {
      name:'Jungle Canopy Level', bpm:120, key:'D major',
      mood:'sunny tropical house · organic groove → disco lift',
      sections:['Sunlit Canopy','Waterfall Grove','Skyline Palms','Moonlit Clearing','Treetop Rush'],
      chords:[[50,54,57,61,64],[47,50,54,57,61],[43,47,50,54,57],[45,50,52,57,59]],
      harmony:['Dmaj9','Bm9','Gmaj9','Asus4/9'],
      labels:{woods:'Slap bass',shaker:'Congas & seeds',keys:'Disco strings',marimba:'Steel drums',kalimba:'Disco arpeggio'},
      description:'Slap bass, congas and bright steel drums grow into four-on-the-floor house, pumping synth bass, filtered disco strings and a sparkling arpeggio.',
    },
    mine: {
      name:'Mine Cart Level', bpm:128, key:'D minor',
      mood:'mechanical funk · rail echoes → acid-house climax',
      sections:['Timber Shaft','Steamworks','Crystal Cavern','Abandoned Foundry','Runaway Rails'],
      chords:[[50,53,57,60,64],[46,50,53,57,60],[43,46,50,53,57],[45,50,52,55,59]],
      harmony:['Dm9','Bbmaj9','Gm9','A7sus4/9'],
      labels:{woods:'Funk guitar',shaker:'Crushed metal',keys:'Machine stabs',marimba:'Rail mallets',kalimba:'Robot vowel hook'},
      description:'Clattering rails, bit-crushed metal and clipped funk guitar lock into a hard electronic kit, resonant acid bass and a synthesized robotic vowel chant.',
    },
    underwater: {
      name:'Underwater Level', bpm:105, key:'D minor',
      mood:'fluid ambient electro · glassy stillness → cinematic surge',
      sections:['Coral Garden','Tidal Grotto','Open Blue','Sunken Temple','Riptide Passage'],
      chords:[[50,53,57,60,64],[46,50,53,57,60],[43,46,50,53,57],[45,50,52,55,59]],
      harmony:['Dm9','Bbmaj9','Gm9','A7sus4/9'],
      labels:{woods:'Deep marimba',shaker:'Bubble rhythm',keys:'Cinematic strings',marimba:'Glass marimba',kalimba:'Sonar arpeggio'},
      description:'Glassy pads, echoing breathy flutes and deep marimbas float above underwater ambience, then open into a sidechained kick and cinematic analog bass.',
    },
    orbital: {
      name:'Orbital Rainforest', bpm:112, key:'D minor', biome:true, reverb:4.2,
      mood:'luminous, curious · alien calls become the groove',
      sections:['Spore Landing','Lunar Falls','Ringworld Canopy','Sleeping Monolith','Meteor Bloom'],
      chords:[[50,53,57,60,64],[46,50,53,57,60],[43,46,50,53,57],[45,50,52,55,59]],
      harmony:['Dm9','Bbmaj9','Gm9','A7sus4/9'],
      environment:['Alien rain','crystal drops, spores'],
      labels:{amb:'Luminous haze',water:'Alien rain',wildlife:'Distant creatures',birds:'Canopy replies',woods:'Hollow wood',shaker:'Spore shakers',drums:'Orbital house kit',bass:'Liquid bass',keys:'Cosmic strings',pad:'Warm star pad',marimba:'Creature rhythm',kalimba:'Robotic replies'},
      description:'Alien rainforest calls and crystal rain float over warm celestial pads. Hollow wood, liquid bass and a house pulse draw the creatures into a rhythmic hook, answered by metallic signals at the climax.',
      stages:['Distant, irregular creature calls, luminous haze and crystal rain.','The same creature timbre locks to the groove as wood percussion and liquid bass arrive.','Creature hooks trade answers with metallic signals beneath soaring cosmic strings.'],
    },
    crystal: {
      name:'Crystal Caverns', bpm:100, key:'E minor', biome:true, reverb:5.4,
      mood:'resonant, subterranean · stone echoes become melodic techno',
      sections:['Quartz Entrance','Dripstone Pool','Amethyst Vault','Obsidian Chamber','Prismatic Heart'],
      chords:[[52,55,59,62,66],[48,52,55,59,62],[45,48,52,55,59],[47,52,54,57,62]],
      harmony:['Em9','Cmaj9','Am9','Bm11'],
      environment:['Seepage','drips, mineral pools'],
      labels:{amb:'Bowed glass',water:'Mineral drips',wildlife:'Cavern resonance',birds:'Distant crystals',woods:'Stone mallets',shaker:'Shale grains',drums:'Deep cave pulse',bass:'Subterranean bass',keys:'Prismatic choir',pad:'Geode pad',marimba:'Crystal melody',kalimba:'Quartz sequencer'},
      description:'Bowed glass, tuned water drops and slowly ringing mineral tones unfold into deep melodic techno, with subterranean bass, stone percussion and a bright quartz sequence.',
      stages:['Slow bowed glass, distant mineral tones and individual water drops.','A deep pulse and stone mallets reveal the cavern’s rhythm.','A quartz sequence and prismatic harmonies bloom over the full techno pulse.'],
    },
    greenhouse: {
      name:'Clockwork Greenhouse', bpm:125, key:'G major', biome:true, reverb:2.1,
      mood:'intricate, playful · ticking mechanisms become botanical funk',
      sections:['Seedling Atrium','Irrigation Works','Glass Conservatory','Overgrown Orrery','Clocktower Bloom'],
      chords:[[43,47,50,54,57],[40,43,47,50,54],[48,52,55,59,62],[50,55,57,60,64]],
      harmony:['Gmaj9','Em9','Cmaj9','D7sus4/9'],
      environment:['Irrigation','mist, glass droplets'],
      labels:{amb:'Reed harmonium',water:'Glasshouse mist',wildlife:'Clock mechanisms',birds:'Music-box seeds',woods:'Muted vine guitar',shaker:'Escapement ticks',drums:'Clockwork funk kit',bass:'Plucked synth bass',keys:'Brass-reed stabs',pad:'Sunlit glass pad',marimba:'Porcelain bells',kalimba:'Gear arpeggio'},
      description:'A breathing reed harmonium, music-box seeds and tiny clock mechanisms grow into swung botanical funk: muted guitar, plucked bass, porcelain bells and interlocking gear arpeggios.',
      stages:['Reed harmonium, a delicate music box and scattered clock ticks.','Swung percussion, muted guitar and plucked bass interlock.','Brass-reed stabs and the gear arpeggio set the whole greenhouse dancing.'],
    },
  };
  function random(seed){ return () => { seed = (Math.imul(seed,1664525)+1013904223)>>>0; return seed/4294967296; }; }
  const hz = midi => 440 * 2 ** ((midi-69)/12);
  function energy(state){ return clamp(Math.max(state.intensity || 0, (state.progress || 0)*.65, (state.performance || 0)*.85)); }
  function stage(state){ const e=energy(state); return e < .34 ? 'Exploring' : e < .74 ? 'Action' : 'Climax'; }
  function create(id){
    const config=modes[id]; if(!config) throw new Error('Unknown fusion mode: '+id);
    const beatSamples=RATE*60/config.bpm, n=beatSamples*64, beat=beatSamples/RATE;
    const labels={amb:'Atmospheric bed',water:id==='mine'?'Rail ambience':id==='underwater'?'Ocean currents':'Canopy air',
      wildlife:id==='mine'?'Mechanical echoes':id==='underwater'?'Distant bubbles':'Forest textures',birds:'Breathy flute',
      drums:'909-style kit',bass:id==='mine'?'Acid bass':'Pumping synth bass',pad:'Glassy pad',...config.labels};
    function configure(meta){
      Object.assign(meta,{sr:RATE,bpm:config.bpm,beat:beatSamples,bar:beatSamples*4,loop:n,tail:RATE*9,leadin:{}});
      for (const [i,a] of [...'ABCDE'].entries()){
        meta.stems[a]=[...layers,'amb2','amb3','amb4'];
        meta.sections[a]={name:config.sections[i],mood:config.mood,chords:Array.from({length:16},(_,b)=>config.harmony[Math.floor(b/4)])};
      }
      return meta;
    }
    function arrange(state, arrangement, take){
      const e=energy(state), tier=arrangement==='sparse'?0:arrangement==='medium'?1:arrangement==='full'?2:e<.34?0:e<.74?1:2;
      const t={[take]:.8,water:.3+.5*(state.water || 0),wildlife:.45,birds:tier===2?.85:.4};
      t.drums=tier===0?.08:tier===1?.7:1;
      if(tier>=1) Object.assign(t,{woods:.85,shaker:.65,marimba:.65,bass:.65+.25*e,pad:.65});
      if(tier===2) Object.assign(t,{keys:.85,kalimba:.85,shaker:.9,marimba:.85});
      if(id==='orbital') {
        // Crossfade loose wildlife into a metrically locked version of the same voice.
        t.wildlife=tier===0?.8:tier===1?.3:.12;
        t.birds=tier===0?.3:.55;
        if(tier>=1) t.marimba=tier===1?.75:.95;
      }
      return t;
    }
    function render(ctx,key){
      const [area,stem]=key.split('_');
      if(!'ABCDE'.includes(area) || ![...layers,'amb2','amb3','amb4'].includes(stem)) throw new Error('Unknown stem '+key);
      const variation=area.charCodeAt(0)-65;
      const rnd=random([...id+key].reduce((s,c)=>Math.imul(s,31)+c.charCodeAt(0)|0,19));
      const buffer=ctx.createBuffer(2,n,RATE),t16=ctx.createBuffer(2,RATE*9,RATE),t8=ctx.createBuffer(2,RATE*9,RATE);
      const data=[0,1].map(c=>buffer.getChannelData(c)), end=[0,1].map(c=>t16.getChannelData(c)),half=[0,1].map(c=>t8.getChannelData(c));
      // A note's pre-boundary samples and its ring-out are derived from the same render.
      function note(b,dur,pitch,amp,pan,voice){
        const start=Math.round(b*beatSamples),len=Math.round(dur*RATE),freq=hz(pitch);
        const gains=[Math.sqrt((1-pan)/2),Math.sqrt((1+pan)/2)];
        let low=0,band=0,lastNoise=0,held=0;
        const formants=b%4<2?[600,1100,2400]:[350,1700,2600];
        const weights=voice==='robot'?Array.from({length:18},(_,i)=>formants.reduce((sum,fc)=>sum+Math.exp(-((((i+1)*freq-fc)/180)**2)),0)):[];
        const phaseStep=2*Math.PI*freq/RATE;
        const attack=voice==='reed'&&dur<1?.015:['pad','strings','nebula','bowed','reed'].includes(voice)?.5:voice==='flute'?.06:.004;
        const release=voice==='reed'&&dur<1?.06:['pad','strings','nebula','bowed','reed'].includes(voice)?1.0:.05;
        for(let j=0;j<len;j++){
          const t=j/RATE,ph=j*phaseStep,noise=rnd()*2-1;
          const envelope=Math.min(1,t/attack)*Math.min(1,(dur-t)/release);
          let v=0;
          if(voice==='pad'||voice==='strings'){
            v=(Math.sin(ph)+.32*Math.sin(ph*2+.5*Math.sin(t*1.4))+.18*Math.sin(ph*3))*.55;
            if(voice==='strings') v += .1*Math.sin(ph*4+.7*Math.sin(t*3));
          } else if(voice==='nebula'){
            v=(Math.sin(ph+.3*Math.sin(t*.9))+.22*Math.sin(ph*2)+.14*Math.sin(ph*3+.6*Math.sin(t*1.7)))*.65;
          } else if(voice==='bowed'){
            v=(Math.sin(ph)+.24*Math.sin(ph*2)+.12*Math.sin(ph*4))*(.7+.2*Math.sin(t*2.1));
          } else if(voice==='reed'){
            v=(Math.sin(ph)+.3*Math.sin(ph*3)+.12*Math.sin(ph*5))*(.75+.1*Math.sin(t*4));
          } else if(voice==='alien'||voice==='signal'){
            const bend=4*(1-Math.exp(-t*12)),trill=.8*Math.sin(t*24);
            v=Math.sin(ph+bend+trill+(voice==='signal'?1.5: .65)*Math.sin(ph*2))*Math.exp(-t*3.2);
          } else if(voice==='geode'||voice==='porcelain'){
            const bell=voice==='geode';
            v=(Math.sin(ph)*Math.exp(-t*.8)+.32*Math.sin(ph*(bell?2.01:3))*Math.exp(-t*2)+.13*Math.sin(ph*(bell?5.02:7))*Math.exp(-t*5))*.65;
          } else if(voice==='gear'){
            v=(noise*.4+Math.sin(ph)*Math.sin(ph*1.7)*.6)*Math.exp(-t*55);
          } else if(voice==='liquid'){
            v=Math.sin(ph+1.4*Math.sin(ph*2)*Math.exp(-t*7))*Math.exp(-t*2.2)*.8;
          } else if(voice==='sub'){
            v=(Math.sin(ph)+.18*Math.sin(ph*2))*Math.exp(-t*1.4);
          } else if(voice==='flute'){
            v=(Math.sin(ph+.016*Math.sin(t*31))+.18*Math.sin(ph*2)+noise*.05)*Math.exp(-t*.65);
          } else if(voice==='kick'){
            v=Math.sin(2*Math.PI*48*t+10*(1-Math.exp(-t*38)))*Math.exp(-t*9)+noise*.12*Math.exp(-t*110);
          } else if(voice==='snare'){
            v=(noise*.7+Math.sin(2*Math.PI*180*t)*.3)*Math.exp(-t*19);
          } else if(voice==='hat'||voice==='metal'){
            const bright=noise-lastNoise; lastNoise=noise;
            if(voice==='metal') { if(j%5===0) held=Math.round((.4*bright+.4*Math.sin(ph)*Math.sin(ph*1.47))*16)/16; v=held*Math.exp(-t*16); }
            else v=bright*.55*Math.exp(-t*35);
          } else if(voice==='acid'){
            const saw=2*((freq*t)%1)-1;
            const cutoff=300+2200*Math.exp(-t*9),f=2*Math.sin(Math.PI*cutoff/RATE);
            low+=f*band; const high=saw-low-.48*band; band+=f*high;
            v=Math.tanh(low*2.2)*.6*Math.exp(-t*2);
          } else if(voice==='analog'){
            v=(Math.sin(ph)+.32*Math.sin(ph*2)+.16*Math.sin(ph*3)+.07*Math.sin(ph*5))*Math.exp(-t*2)*.7;
          } else if(voice==='slap'||voice==='guitar'){
            v=(Math.sin(ph)+.42*Math.sin(ph*2)+.22*Math.sin(ph*4))*Math.exp(-t*(voice==='slap'?6:12));
            v+=noise*.15*Math.exp(-t*90); v=Math.tanh(v*1.4)*.7;
          } else if(voice==='robot'){
            // Harmonic formant synthesis: vowel colours, no recorded or imitated voice.
            for(let h=1;h<=18;h++){
              v+=Math.sin(ph*h)*weights[h-1]*.22;
            }
            v*=.65+.35*Math.sin(Math.PI*t/dur);
          } else if(voice==='air'){
            low+=(noise-low)*.025; v=low*3*(.65+.35*Math.sin(t*2));
          } else if(voice==='bubble'){
            v=Math.sin(ph+10*(1-Math.exp(-t*8)))*Math.exp(-t*8);
          } else if(voice==='conga'){
            v=(Math.sin(ph+2*Math.exp(-t*28))+.2*Math.sin(ph*1.6))*Math.exp(-t*14);
          } else {
            const metal=voice==='steel'?2.76:3.99;
            v=(Math.sin(ph)+.45*Math.sin(ph*metal)*Math.exp(-t*8)) * Math.exp(-t*(voice==='steel'?2.7:4.5));
          }
          v*=amp*envelope;
          const pos=start+j;
          for(let c=0;c<2;c++){
            const sample=v*gains[c]; data[c][pos%n]+=sample;
            if(pos>=n&&pos-n<end[c].length) end[c][pos-n]+=sample;
            if(start<n/2&&pos>=n/2&&pos-n/2<half[c].length) half[c][pos-n/2]+=sample;
          }
        }
      }
      const chordAt=b=>config.chords[Math.floor(b/16)%4];
      const motif=[0,2,3,1,2,4,3,1];
      if(config.biome) {
        const orbital=id==='orbital', crystal=id==='crystal', green=id==='greenhouse';
        const tint=orbital?'nebula':crystal?'bowed':'reed';
        const call=orbital?'alien':crystal?'geode':'porcelain';
        const swing=b=>green && Math.round(b*2)%2===1 ? b+.12 : b;
        const melody=b=>chordAt(b)[motif[(Math.floor(b)+variation)%8]]+12;
        const echo=(b,dur,pitch,amp,voice)=>{
          note(b,dur,pitch,amp,-.3,voice);
          note(b+.75,dur,pitch,amp*.3,.5,voice);
        };
        if(stem.startsWith('amb')||stem==='pad') {
          const ambient=stem.startsWith('amb');
          for(let b=0;b<64;b+=8) chordAt(b).slice(0,ambient?3:5).forEach((pitch,i)=>
            note(b,8*beat+1.5,pitch,ambient?.012:.018,(i%2?1:-1)*(.3+rnd()*.3),tint));
        } else if(stem==='water') {
          if(!crystal) for(let b=0;b<64;b+=8) note(b,8*beat+.5,45,.018,.1,'air');
          for(let b=0;b<64;b+=4) echo(b+.25+rnd(),.85,melody(b)+12,.018,green?'porcelain':'bubble');
        } else if(stem==='wildlife') {
          // Deterministic but deliberately off-grid calls: rhythm stems answer these later.
          for(let b=0;b<62;b+=7) note(b+rnd()*1.7,orbital?1.2:crystal?3:.12,melody(b)+(crystal?-24:0),.035,rnd()*1.4-.7,orbital?'alien':crystal?'geode':'gear');
        } else if(stem==='birds') {
          for(let b=0;b<64;b+=green?4:8) echo(b,crystal?3.2:1.1,melody(b),.036,call);
        } else if(stem==='drums') {
          for(let b=0;b<64;b++) {
            note(b,.6,36,crystal?.17:.19,0,'kick');
            if(b%2===1) note(swing(b),.22,green?69:55,crystal?.045:.062,.12,crystal?'gear':'snare');
            note(swing(b+.5),.1,86,.025,-.2,'hat');
          }
        } else if(stem==='shaker') {
          for(let b=0;b<64;b+=.5) note(swing(b),.14,green?79:crystal?65:88,.031,rnd()-.5,green?'gear':crystal?'metal':'hat');
        } else if(stem==='woods') {
          for(let bar=0;bar<16;bar++) for(const off of (crystal?[0,2.5]:[0,.75,1.5,2.5,3.25])) {
            const b=bar*4+off;
            note(b,.5,chordAt(b)[bar%3],.059,rnd()*.6-.3,green?'guitar':orbital?'conga':'mallet');
          }
        } else if(stem==='bass') {
          for(let b=0;b<64;b+=crystal?2:1) note(swing(b+(green?.5:0)),(crystal?1.7:.75)*beat,chordAt(b)[b%4===3?2:0]-12,.09,0,orbital?'liquid':crystal?'sub':'slap');
        } else if(stem==='keys') {
          for(let b=0;b<64;b+=green?2:8) chordAt(b).forEach((pitch,i)=>
            note(b+(green?.5:0),green?.4:8*beat+1.5,pitch+12,green?.026:.018,(i%2?1:-1)*.5,green?'reed':crystal?'bowed':'strings'));
        } else if(stem==='marimba') {
          for(let b=0;b<64;b+=orbital?.5:crystal?1.5:1) echo(swing(b),orbital?.3:1.4,melody(b),orbital?.046:.041,call);
        } else if(stem==='kalimba') {
          for(let b=0;b<64;b+=orbital?2:.5) note(swing(b+.25),orbital?.45:1,chordAt(b)[Math.floor(b*2+variation)%5]+12,.033,.3,orbital?'signal':call);
        }
        return {buffer,tails:{t16,t8}};
      }
      if(stem.startsWith('amb')||stem==='pad'||stem==='keys'){
        const ambient=stem.startsWith('amb');
        for(let b=0;b<64;b+=4){
          const chord=chordAt(b),voice=stem==='keys'?(id==='mine'?'guitar':'strings'):'pad';
          chord.slice(0,ambient?3:5).forEach((pitch,i)=>note(b,stem==='keys'&&id==='mine'?.5:4*beat+1.3,pitch+(stem==='keys'?12:0),ambient?.014:.019,(i%2?1:-1)*(.25+rnd()*.4),voice));
        }
      } else if(stem==='water') {
        for(let b=0;b<64;b+=4) note(b,4*beat+.6,45,.025,rnd()-.5,'air');
      } else if(stem==='wildlife') {
        for(let b=0;b<64;b+=2+variation%2) note(b,.5,chordAt(b)[b%5]+12,.018,rnd()*1.6-.8,id==='mine'?'metal':id==='underwater'?'bubble':'flute');
      } else if(stem==='birds') {
        for(let b=0;b<64;b+=4){
          const pitch=chordAt(b)[motif[(b/4+variation)%8]]+12;
          note(b,1.6,pitch,.042,-.25,'flute');
          note(b+.75,1.6,pitch,.016,.5,'flute');
        }
      } else if(stem==='drums') {
        for(let b=0;b<64;b++){
          note(b,.6,36,.2,0,'kick');
          if(b%2===1) note(b,.28,48,.075,.1,'snare');
          note(b+.5,.14,86,.033,-.25,'hat');
        }
      } else if(stem==='shaker') {
        for(let b=0;b<64;b+=.5) note(b,.24,55+(b%2)*7,.045,rnd()-.5,id==='mine'?'metal':id==='canopy'?'conga':'bubble');
      } else if(stem==='woods') {
        for(let b=0;b<64;b+=.75){
          const pitch=chordAt(b)[Math.floor(b)%3]+(id==='canopy'?-12:0);
          note(b,.45,pitch,.075,rnd()*.5-.25,id==='canopy'?'slap':id==='mine'?'guitar':'mallet');
        }
      } else if(stem==='bass') {
        for(let b=0;b<64;b+=id==='mine'?.5:1){
          const pitch=chordAt(b)[b%4===3?2:0]-12;
          note(b,beat*.8,pitch,.10,0,id==='mine'?'acid':'analog');
        }
      } else if(stem==='marimba') {
        for(let b=0;b<64;b+=id==='underwater'?2:1){
          note(b,1.3,chordAt(b)[motif[(Math.floor(b)+variation)%8]]+12,.06,rnd()*.8-.4,id==='canopy'?'steel':'mallet');
        }
      } else if(stem==='kalimba') {
        for(let b=0;b<64;b+=id==='mine'?2:.5){
          note(b,id==='mine'?.5:.45,chordAt(b)[Math.floor(b*2)%5]+(id==='mine'?0:12),id==='mine'?.075:.038,.25,id==='mine'?'robot':'steel');
        }
      }
      return {buffer,tails:{t16,t8}};
    }
    return {config,labels,configure,arrange,render,energy,stage};
  }
  function frame(id,canvas,glow,time,intensity,area='A'){
    const g=canvas.getContext('2d'); glow.getContext('2d').clearRect(0,0,320,140);
    const rnd=random(674);
    const zone=Math.max(0,'ABCDE'.indexOf(area));
    if(id==='orbital'){
      const sky=g.createLinearGradient(0,0,0,140);sky.addColorStop(0,'#090d2e');sky.addColorStop(1,'#193b45');g.fillStyle=sky;g.fillRect(0,0,320,140);
      for(let i=0;i<65;i++){g.fillStyle=i%4?'#91abc6':'#deeddd';g.globalAlpha=.3+.4*Math.sin(time*.4+i)**2;g.fillRect(rnd()*320,rnd()*75,1,1);}g.globalAlpha=1;
      g.fillStyle=['#8e639f','#668ec0','#ba947c','#666eab','#b4748f'][zone];g.beginPath();g.arc(245,39,23,0,Math.PI*2);g.fill();
      g.fillStyle='#152442';g.beginPath();g.arc(253,35,22,0,Math.PI*2);g.fill();g.strokeStyle='#8bcac0';g.beginPath();g.ellipse(245,39,36,5,-.25,0,Math.PI*2);g.stroke();
      for(let depth=0;depth<2;depth++)for(let i=0;i<7;i++){
        const x=i*53+(depth?14:0),y=depth?52:43;
        g.fillStyle=depth?'#163b41':'#17313f';g.fillRect(x,y,5,100);
        g.fillStyle=depth?'#265653':'#1c4048';g.fillRect(x-15,y-9,35,11);g.fillRect(x-8,y-16,23,10);
        g.fillStyle=depth?'#62c8ae':'#497c84';g.fillRect(x-14,y,33,2);
        g.strokeStyle='#407069';g.beginPath();g.moveTo(x+12,y+2);g.bezierCurveTo(x+22,y+30,x+3,y+22,x+12,y+60);g.stroke();
      }
      g.fillStyle='#112b35';g.fillRect(0,124,320,16);
      for(let i=0;i<24;i++){
        const x=rnd()*320,y=117+rnd()*17;g.fillStyle=i%2?'#6dcfc7':'#b08de0';g.fillRect(x,y-4,5,2);g.fillRect(x+2,y-2,1,5);
      }
      for(let i=0;i<25;i++){
        const x=(rnd()*320+Math.sin(time*.4+i)*8+320)%320,y=(rnd()*130-time*(.4+intensity)+1400)%140;
        const pulse=.4+.6*Math.sin(time*(1+intensity*2)+i)**2;
        g.fillStyle=`rgba(143,255,203,${pulse})`;g.fillRect(x,y,1,2);
        const glowG=glow.getContext('2d');glowG.fillStyle=`rgba(84,234,183,${pulse*.3})`;glowG.fillRect(x-1,y-1,3,3);
      }
    }else if(id==='crystal'){
      const sky=g.createLinearGradient(0,0,0,140);sky.addColorStop(0,'#111328');sky.addColorStop(1,'#29395b');g.fillStyle=sky;g.fillRect(0,0,320,140);
      const colours=[['#415b88','#93d5ee'],['#486078','#9ce0dd'],['#635083','#c8a8f1'],['#40455f','#969fcd'],['#477586','#b6f6de']][zone];
      for(let i=0;i<16;i++){const x=i*24,h=12+rnd()*40;g.fillStyle='#1a2037';g.beginPath();g.moveTo(x,0);g.lineTo(x+12,h);g.lineTo(x+24,0);g.fill();}
      g.fillStyle='#162840';g.fillRect(0,111,320,29);
      for(let i=0;i<12;i++){
        const x=i*28-10,h=18+rnd()*47,y=118+rnd()*10;
        g.fillStyle=colours[0];g.beginPath();g.moveTo(x,y);g.lineTo(x+3,y-h);g.lineTo(x+11,y-h-7);g.lineTo(x+22,y-h+2);g.lineTo(x+24,y);g.fill();
        g.fillStyle=colours[1];g.globalAlpha=.35+.2*Math.sin(time*.8+i)**2;g.beginPath();g.moveTo(x+11,y-h-7);g.lineTo(x+13,y);g.lineTo(x+20,y);g.lineTo(x+22,y-h+2);g.fill();g.globalAlpha=1;
        g.fillStyle='#d6f1ff';g.fillRect(x+11,y-h-5,1,3);
      }
      g.fillStyle='#102739';g.fillRect(77,116,164,24);
      for(let i=0;i<9;i++){g.fillStyle=i%2?'#528d9d':'#2c526f';const x=85+rnd()*149;g.fillRect(x+Math.sin(time*.6+i)*3,120+i*2,8+rnd()*18,1);}
      for(let i=0;i<5;i++){const x=95+i*32,y=(time*(12+intensity*6)+i*19)%118;g.fillStyle='#a9d5e5';g.fillRect(x,y,1,2);}
      const glowG=glow.getContext('2d');glowG.fillStyle='rgba(94,139,226,.35)';glowG.fillRect(20,99,270,15);
    }else if(id==='greenhouse'){
      const sky=g.createLinearGradient(0,0,0,140);sky.addColorStop(0,['#4c827e','#597d88','#7c9186','#506c6e','#a18c6d'][zone]);sky.addColorStop(1,'#cfbd8a');g.fillStyle=sky;g.fillRect(0,0,320,140);
      g.fillStyle='rgba(219,236,198,.18)';for(let i=0;i<5;i++)g.fillRect(i*66+5,10,55,98);
      g.strokeStyle='#3b5b59';g.lineWidth=2;
      for(let i=0;i<6;i++){g.beginPath();g.moveTo(160,-18);g.lineTo(i*64,114);g.stroke();}for(const y of [29,63,99]){g.beginPath();g.moveTo(0,y);g.lineTo(320,y);g.stroke();}
      function gear(x,y,r,phase){
        g.save();g.translate(x,y);g.rotate(phase);g.fillStyle='#8c774c';g.beginPath();g.arc(0,0,r-2,0,Math.PI*2);g.fill();
        for(let j=0;j<12;j++){g.rotate(Math.PI/6);g.fillRect(r-4,-2,6,4);}g.fillStyle='#354f4a';g.beginPath();g.arc(0,0,r*.5,0,Math.PI*2);g.fill();g.fillStyle='#c8b87c';g.fillRect(-2,-2,4,4);g.restore();
      }
      gear(249,66,19,time*(.12+intensity*.5));gear(277,86,14,-time*(.17+intensity*.65));gear(59,52,12,-time*.2);
      g.fillStyle='#536650';g.fillRect(0,124,320,16);g.fillStyle='#9e8159';g.fillRect(13,111,294,5);
      for(let i=0;i<10;i++){
        const x=19+i*31,h=15+rnd()*24;g.fillStyle='#9b7156';g.fillRect(x,104,13,8);g.fillStyle='#416a48';g.fillRect(x+6,104-h,2,h);
        for(let j=0;j<3;j++){g.fillStyle=j%2?'#779558':'#49754d';g.fillRect(x+(j%2?7:0),98-j*7,7,3);}
        g.fillStyle=i%3?'#ceb980':'#d99e79';g.fillRect(x+4,103-h,6,3);
      }
      g.strokeStyle='#baaa6f';g.lineWidth=1;g.beginPath();g.moveTo(0,90);g.lineTo(320,90);g.stroke();
      for(let i=0;i<18;i++){g.fillStyle='rgba(229,241,204,.5)';g.fillRect(rnd()*320,(rnd()*120+time*2)%120,1,1);}
    }else if(id==='underwater'){
      const grad=g.createLinearGradient(0,0,0,140);grad.addColorStop(0,'#146878');grad.addColorStop(1,'#071b35');g.fillStyle=grad;g.fillRect(0,0,320,140);
      for(let i=0;i<8;i++){g.fillStyle='rgba(124,232,240,.06)';g.beginPath();g.moveTo(i*48+Math.sin(time*.2+i)*8,0);g.lineTo(i*48-30,140);g.lineTo(i*48+15,140);g.fill();}
      for(let i=0;i<35;i++){const x=rnd()*320,y=140-((rnd()*140+time*(2+intensity*3))%140);g.strokeStyle='#65b6c8';g.strokeRect(x,y,2,2);}
      for(let i=0;i<22;i++){const x=i*16;g.fillStyle=i%2?'#936d9d':'#247d81';g.fillRect(x,128-rnd()*16,3,20);g.fillRect(x-3,125,9,3);}
      const x=140+Math.sin(time*.4)*30;g.fillStyle='#edd098';g.fillRect(x,72,12,5);g.fillRect(x+12,70,3,9);
    }else{
      g.fillStyle='#151322';g.fillRect(0,0,320,140);
      for(let i=0;i<8;i++){const x=((i*48-time*(8+intensity*24))%384+384)%384-32;g.fillStyle='#493543';g.fillRect(x,10,7,110);g.fillRect(x,10,48,7);g.fillStyle='#e6b45a';g.fillRect(x+8,28,3,6);}
      g.fillStyle='#745766';g.fillRect(0,116,320,3);g.fillRect(0,127,320,3);
      for(let i=0;i<24;i++){g.fillStyle='#9a7580';g.fillRect(((i*16-time*20)%384+384)%384-32,119,3,8);}
      const y=99+Math.round(Math.sin(time*12)*intensity);g.fillStyle='#a9a5bd';g.fillRect(126,y,28,12);g.fillStyle='#c99358';g.fillRect(129,y-7,21,7);g.fillStyle='#252536';g.fillRect(130,y+12,5,5);g.fillRect(145,y+12,5,5);
      for(let i=0;i<12;i++){g.fillStyle='#58a6b8';g.fillRect(rnd()*320,40+rnd()*35,2,6);}
    }
  }
  return {modes,create,frame,energy,stage};
})();
