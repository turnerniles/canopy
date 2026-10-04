/* Render off the audio/UI thread, then transfer each array without copying. */
'use strict';
importScripts('fusion.js');
self.onmessage = ({data:{mode,key}}) => {
  try {
    const ctx={createBuffer(channels,length,sampleRate){
      const data=Array.from({length:channels},()=>new Float32Array(length));
      return {sampleRate,getChannelData:c=>data[c]};
    }};
    const result=CanopyFusion.create(mode).render(ctx,key);
    const buffers=[result.buffer,result.tails.t16,result.tails.t8];
    const arrays=buffers.flatMap(b=>[b.getChannelData(0),b.getChannelData(1)]);
    self.postMessage({key,rate:result.buffer.sampleRate,arrays}, arrays.map(a=>a.buffer));
  } catch(error){ self.postMessage({key,error:error.message}); }
};
