const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
require('./fusion.js');
const ctx = {createBuffer(channels,length,sampleRate) {
  const data=Array.from({length:channels},()=>new Float32Array(length));
  return {length,sampleRate,getChannelData:c=>data[c]};
}};
let count=0,maxPeak=0,maxSeam=0;
for(const mode of Object.keys(CanopyFusion.modes)) {
  assert.ok(fs.readFileSync(path.join(__dirname,'index.html'),'utf8').includes('value="'+mode+'"'), mode+' in soundtrack selector');
  const score=CanopyFusion.create(mode);
  const meta=score.configure(JSON.parse(fs.readFileSync(path.join(__dirname,'meta.json'),'utf8')));
  assert.equal(meta.beat,Math.round(meta.beat));
  assert.equal(meta.loop,meta.bar*16);
  assert.equal(meta.loop/meta.sr,64*60/meta.bpm);
  const low=score.arrange({intensity:.1},'breathe','amb');
  const medium=score.arrange({intensity:.5},'breathe','amb');
  const high=score.arrange({intensity:.95},'breathe','amb');
  assert.ok(low.drums < .1 && !low.bass && !low.keys);
  assert.ok(medium.bass > 0 && medium.woods > 0 && !medium.kalimba);
  assert.ok(high.kalimba > 0 && high.keys > 0 && high.drums === 1);
  if(mode==='orbital') {
    assert.ok(!low.marimba && medium.marimba > 0 && high.marimba > medium.marimba, 'creature rhythm enters and grows');
    assert.ok(low.wildlife > medium.wildlife && medium.wildlife > high.wildlife, 'distant creatures recede as rhythmic calls enter');
  }
  assert.equal(score.stage({intensity:0,performance:1}),'Climax');
  assert.equal(score.stage({intensity:0,progress:1}),'Action');
  assert.ok(!score.arrange({intensity:1},'sparse','amb').bass);
  for(const area of 'ABCDE') {
    const mix=new Float32Array(meta.loop);
    for(const stem of meta.stems[area]) {
      const {buffer,tails}=score.render(ctx,area+'_'+stem);
      assert.equal(buffer.length,meta.loop);
      assert.equal(buffer.sampleRate,meta.sr);
      assert.equal(tails.t16.length,meta.tail);
      assert.equal(tails.t8.length,meta.tail);
      for(let c=0;c<2;c++) {
        const data=buffer.getChannelData(c);
        let energy=0;
        for(let i=0;i<data.length;i++) {
          const v=data[i]; assert.ok(Number.isFinite(v),`${mode}/${area}/${stem} finite`);
          energy+=v*v; maxPeak=Math.max(maxPeak,Math.abs(v));
          if(c===0 && !['amb2','amb3','amb4'].includes(stem)) mix[i]+=v;
        }
        assert.ok(energy>.001,`${mode}/${area}/${stem} audible`);
        const seam=Math.abs(data[0]-data[data.length-1]); maxSeam=Math.max(seam,maxSeam);
        assert.ok(seam<.035,`${mode}/${area}/${stem} seam ${seam}`);
        assert.ok(Math.abs(data[0]-tails.t16.getChannelData(c)[0])<1e-6, 'wrapped release matches tail');
        assert.ok(Math.abs(data[meta.loop/2]-tails.t8.getChannelData(c)[0])<1e-5, 'half-loop release matches tail');
      }
      count++;
    }
    for(const v of mix) assert.ok(Math.abs(v)<.95,`${mode}/${area} mix headroom`);
  }
  const a=score.render(ctx,'A_birds').buffer.getChannelData(0);
  const b=score.render(ctx,'A_birds').buffer.getChannelData(0);
  assert.deepEqual(a,b);
  console.log(`${mode}: exact grid, gameplay builds, 75 audio stems and full-mix headroom passed`);
}
console.log(`${count} stems passed; peak ${maxPeak.toFixed(4)}, seam ${maxSeam.toFixed(4)}`);
