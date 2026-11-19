// Offline behavioral checks; no browser packages or running web server required.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dir = path.dirname(fileURLToPath(import.meta.url));
class Element {
  constructor() { this.value=''; this.children=[]; this.dataset={}; this.style={}; this.events={}; this.attributes={}; this.checked=false; }
  append(...els) { this.children.push(...els); }
  add(el) { this.children.push(el); if (!this.value) this.value=el.value; }
  addEventListener(type,fn) { this.events[type]=fn; }
  setAttribute(key,value) { this.attributes[key]=value; }
  replaceChildren(...els) { this.children=els; this.value=''; }
  click() { if (this.disabled) return; this.events.click?.(); this.onclick?.(); }
}
const nodes = new Map();
const get = id => { if (!nodes.has(id)) nodes.set(id,new Element()); return nodes.get(id); };
for (const [id,value] of Object.entries({volume:'35',speed:'1',size:'85','sound-category':'emotions'})) get(id).value=value;
get('auto-blink').checked=false;
const draws=[];
get('character').getContext = () => ({clearRect(){draws.length=0;},save(){},restore(){},translate(){},drawImage(img){draws.push(img.src);}});
const docEvents={},winEvents={},rafs=new Map(),audios=[];
let now=0, nextRaf=1;
const document={hidden:false,getElementById:get,createElement:()=>new Element(),addEventListener:(k,v)=>docEvents[k]=v,querySelectorAll:()=>[...get('motions').children,get('apply-face'),get('pause')]};
class TestImage { set src(s){this._src=s; queueMicrotask(()=>this.onload());} get src(){return this._src;} }
class TestAudio {
  constructor(src){this.src=src; this.paused=true; this.events={}; audios.push(this);}
  addEventListener(k,v){this.events[k]=v;}
  async play(){this.paused=false;}
  pause(){this.paused=true;}
}
const window={addEventListener:(k,v)=>winEvents[k]=v};
const fixedMath=Object.create(Math);fixedMath.random=()=>0.5;
const sandbox={Math:fixedMath,window,document,Image:TestImage,Audio:TestAudio,Option:class{constructor(text,value){this.text=text;this.value=value;}},performance:{now:()=>now},requestAnimationFrame:fn=>{const id=nextRaf++;rafs.set(id,fn);return id;},cancelAnimationFrame:id=>rafs.delete(id),console};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(dir,'catalog.js'),'utf8'),sandbox);
vm.runInContext(fs.readFileSync(path.join(dir,'preview.js'),'utf8'),sandbox);
const flush=async()=>{for(let i=0;i<8;i++)await Promise.resolve();};
function step(ms=50){now+=ms;const callbacks=[...rafs.values()];rafs.clear();callbacks.forEach(fn=>fn(now));}
function motion(name){get('motions').children.find(b=>b.dataset.motion===name).click();}
function emotion(name){get('expressions').children.find(b=>b.dataset.expression===name).click();}
function advance(ms){for(let i=0;i<ms/50;i++)step();}
await flush();
assert.match(get('catalog-status').textContent,/84 PNG/);
assert.equal(rafs.size,1);
assert.equal(audios.length,0);
console.log('PASS: packs load; no automatic audio');
get('auto-blink').checked=false;
for(const name of ['nod','shake','blink']){motion(name);advance(1600);assert.equal(get('motions').children.find(b=>b.attributes['aria-pressed']==='true').dataset.motion,'idle');}
console.log('PASS: all one-shot gestures return to idle');
motion('talk');advance(2000);assert.equal(get('state-label').textContent,'Разговор');
get('pause').click();const frame=get('timeline').value;advance(500);assert.equal(get('timeline').value,frame);
get('timeline').value=3;get('timeline').oninput();assert.equal(get('time-label').textContent,'4 / 8');
console.log('PASS: talk loops; pause and frame seeking hold');
emotion('sad');assert.equal(get('state-label').textContent,'Разговор');motion('idle');get('pause').click();get('timeline').value=0;get('timeline').oninput();assert.ok(draws.at(-1).endsWith('sprites/expressions/sad.png'));
motion('blink');get('timeline').value=2;get('timeline').oninput();assert.ok(draws.some(x=>x.endsWith('parts/eyes/closed.png')));assert.ok(draws.some(x=>x.endsWith('parts/mouths/frown.png')));
console.log('PASS: blinking preserves selected expression mouth');
get('eyes').value='look_right';get('mouth').value='grin';get('apply-face').click();get('pause').click();get('timeline').value=0;get('timeline').oninput();assert.ok(draws.some(x=>x.endsWith('parts/eyes/look_right.png')));assert.ok(draws.some(x=>x.endsWith('parts/mouths/grin.png')));
console.log('PASS: independent facial layers render');
get('play-sound').click();await flush();assert.equal(audios.length,1);assert.equal(audios[0].volume,.35);
get('play-sound').click();await flush();assert.equal(audios[0].paused,true);assert.equal(audios[1].paused,false);
get('volume').value=17;get('volume').oninput();assert.equal(audios[1].volume,.17);
motion('idle');assert.equal(audios[1].paused,true);
console.log('PASS: volume, replacement and cancellation of sound');
get('play-sound').click();await flush();document.hidden=true;docEvents.visibilitychange();assert.ok(audios.at(-1).paused);assert.equal(rafs.size,0);
document.hidden=false;docEvents.visibilitychange();assert.equal(rafs.size,1);docEvents.visibilitychange();assert.equal(rafs.size,1);
console.log('PASS: background tab stops audio and has no duplicate animation loops');
get('auto-blink').checked=true;motion('idle');let blinkSeen=false;for(let i=0;i<130;i++){step();blinkSeen ||= get('state-label').textContent==='Моргнуть';}assert.ok(blinkSeen);assert.equal(get('state-label').textContent,'Своё выражение');
console.log('PASS: automatic blinking returns to the custom face');

// Fixed speed is a per-motion policy, never a global setting.
for (const [name, before, after] of [['nod',450,100],['shake',550,100],['blink',550,100]]) {
  get('auto-blink').checked=false; motion(name); step(); advance(before);
  assert.equal(get('motions').children.find(b=>b.attributes['aria-pressed']==='true').dataset.motion,name);
  assert.equal(get('speed').textContent, name==='blink'?'1×':'2×');
  advance(after); assert.equal(get('state-label').textContent,'Своё выражение');
  assert.equal(get('speed').textContent,'1×');
}
console.log('PASS: nod/shake are fixed at 2x, blink/idle at 1x');
emotion('neutral');motion('idle');get('pause').click();
for(let frame=0;frame<12;frame++) {
  get('timeline').value=frame;get('timeline').oninput();
  assert.ok(draws.at(-1).endsWith('sprites/expressions/neutral.png'));
  assert.ok(!draws.some(x=>x.includes('animations/idle/')||x.includes('eyes/closed')));
}
console.log('PASS: every idle frame uses an unblinking face');
get('reaction-sound').checked=true;
for(const name of ['talk','shake']) { now+=1000;const count=audios.length;motion(name);assert.equal(audios.length,count); }
get('sound-category').value='character';get('sound-category').onchange();assert.ok(!get('sound').children.some(x=>x.value.startsWith('character/talk_')||x.value==='character/beep_no'));
get('sound-category').value='motion';get('sound-category').onchange();assert.ok(!get('sound').children.some(x=>x.value==='motion/shake_head'));
console.log('PASS: talk/no sound effects are neither bound nor selectable');
