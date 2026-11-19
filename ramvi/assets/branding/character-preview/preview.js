'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const data = window.RAMVI_PACK;
  if (!data) { $('catalog-status').textContent = 'Не найден catalog.js. Соберите каталог командой из README.'; return; }
  const pack = data.character;
  const imageRoot = '../RAMVI_Character_Pack/';
  const soundRoot = '../RAMVI_Character_Sound_Pack/';
  const ctx = $('character').getContext('2d');
  const labels = {neutral:'Спокойствие', confident:'Уверенность', cheerful:'Радость', laugh:'Смех', surprise:'Удивление', wink:'Подмигивание', angry:'Злость', sad:'Грусть', sleepy:'Сонливость', thinking:'Размышление', talking:'Речь', excited:'Восторг', side_eye:'Косой взгляд', idle:'Покой', blink:'Моргнуть', talk:'Разговор', nod:'Кивнуть', shake:'Сказать «нет»', front:'Анфас', three_quarter_left:'↖ Влево', profile_left:'← Влево сильнее', three_quarter_right:'Вправо ↗', profile_right:'Сильнее вправо →', normal:'Обычные', half:'Прикрытые', closed:'Закрытые', happy:'Радостные', surprised:'Удивлённые', look_left:'Взгляд влево', look_right:'Взгляд вправо', sleep:'Сон', smile:'Улыбка', open:'Открытый', wide:'Широкая улыбка', o:'Округлый', grin:'Усмешка', frown:'Недовольный', tiny:'Маленький', talk_a:'Речь · А', talk_e:'Речь · Э'};
  const images = new Map();
  const motionSpeed = name => ['nod','shake'].includes(name) ? 2 : 1;
  const usableSound = item => !item.id.startsWith('character/talk_') && !['motion/shake_head','character/beep_no'].includes(item.id);
  const sounds = data.sounds.files.filter(usableSound);
  let mode = 'idle', expression = 'neutral', custom = false, turn = null;
  let paused = false, elapsed = 0, previous = null, ready = false;
  let blinkAt = 3000 + Math.random() * 3000, quietTime = 0;
  let audio = null, audioSerial = 0, lastReaction = -Infinity, raf = null;
  const notice = text => { $('notice').textContent = text; };
  const key = path => path.split('/').pop().replace('.png','');
  function makeButton(text, action, parent, extra = '') {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = text;
    if (extra) b.className = extra;
    b.addEventListener('click', action); parent.append(b); return b;
  }
  function stopAudio() {
    audioSerial++;
    if (audio) { audio.pause(); audio.currentTime = 0; audio = null; }
    $('play-sound').textContent = '▷ Послушать';
  }
  async function playSound(item) {
    stopAudio();
    if (!item) return;
    const serial = audioSerial;
    const current = new Audio(soundRoot + item.file); audio = current;
    current.volume = Number($('volume').value) / 100;
    current.addEventListener('ended', () => { if (audio === current) { audio = null; $('play-sound').textContent = '▷ Послушать'; } });
    current.addEventListener('error', () => { if (serial === audioSerial) { stopAudio(); notice('Не удалось прочитать звук: ' + item.file); } });
    try {
      await current.play();
      if (serial !== audioSerial) { current.pause(); return; }
      $('play-sound').textContent = '♪ Воспроизводится';
      notice(item.description_ru + ' · ' + item.duration_s.toFixed(2) + ' с');
    } catch (error) {
      if (serial === audioSerial) { stopAudio(); notice('Звук не запущен. Нажмите «Послушать» ещё раз.'); }
    }
  }
  function reaction(name) {
    if (!$('reaction-sound').checked || performance.now() - lastReaction < 700 || name === 'idle') return;
    const ids = {blink:'motion/blink_01', nod:'motion/nod'};
    const item = sounds.find(x => x.id === ids[name]);
    if (item) { lastReaction = performance.now(); playSound(item); }
  }
  const motionButtons = {};
  for (const name of Object.keys(pack.animations)) {
    motionButtons[name] = makeButton(labels[name], () => { playMotion(name); reaction(name); }, $('motions'));
    motionButtons[name].dataset.motion = name;
  }
  const expressionButtons = {};
  for (const [name, value] of Object.entries(pack.expressions)) {
    const b = makeButton('', () => {
      expression = name; custom = false; turn = null;
      $('eyes').value = value.eyes; $('mouth').value = value.mouth;
      syncControls(); draw();
    }, $('expressions'), 'expression');
    b.dataset.expression = name;
    const img = document.createElement('img'); img.src = imageRoot + value.file; img.alt = ''; img.loading = 'lazy';
    const span = document.createElement('span'); span.textContent = labels[name] || name;
    b.append(img, span); expressionButtons[name] = b;
  }
  const turnButtons = {};
  for (const [name, path] of Object.entries(pack.turns)) {
    turnButtons[name] = makeButton(labels[name] || name, () => {
      turn = name === 'front' ? null : path; custom = false; expression = 'neutral';
      playMotion('idle');
      if (turn) notice('Поворот — 2D-деформация. Моргание и жесты доступны после возврата в анфас.');
    }, $('turns'));
    turnButtons[name].dataset.turn = name;
  }
  for (const [id, paths] of [['eyes',pack.eye_parts], ['mouth',pack.mouth_parts]]) {
    for (const path of paths) { const opt = new Option(labels[key(path)] || key(path), key(path)); $(id).add(opt); }
  }
  $('eyes').value = 'normal'; $('mouth').value = 'neutral';
  $('apply-face').onclick = () => { custom = true; turn = null; syncControls(); draw(); };
  $('reset-face').onclick = () => { expressionButtons.neutral.click(); };
  function playMotion(name) {
    stopAudio();
    mode = name; elapsed = 0; quietTime = 0; paused = false; previous = null;
    if (name !== 'idle') turn = null;
    syncControls();
  }
  function syncControls() {
    for (const [name,b] of Object.entries(motionButtons)) b.setAttribute('aria-pressed', String(name === mode));
    for (const [name,b] of Object.entries(expressionButtons)) b.setAttribute('aria-pressed', String(!custom && !turn && name === expression));
    for (const [name,b] of Object.entries(turnButtons)) b.setAttribute('aria-pressed', String(turn ? pack.turns[name] === turn : name === 'front'));
    $('pause').textContent = paused ? '▷' : 'Ⅱ';
    $('pause').setAttribute('aria-label', paused ? 'Продолжить' : 'Приостановить');
    const frames = pack.animations[mode].frames.length;
    $('timeline').max = frames - 1;
    $('speed').textContent = motionSpeed(mode) + '×';
    $('state-label').textContent = mode === 'idle' ? (turn ? 'Поворот' : custom ? 'Своё выражение' : labels[expression]) : labels[mode];
  }
  $('pause').onclick = () => { paused = !paused; previous = null; syncControls(); };
  $('timeline').oninput = () => { elapsed = Number($('timeline').value) * pack.animations[mode].duration_ms; paused = true; previous = null; syncControls(); draw(); };
  $('background').onchange = () => { $('stage').dataset.background = $('background').value; };
  $('size').oninput = () => { $('character').style.width = $('size').value + '%'; };
  function face(eyes, mouth, offset = 0) {
    for (const path of [pack.character_base, 'parts/eyes/' + eyes + '.png', 'parts/mouths/' + mouth + '.png']) ctx.drawImage(images.get(path),0,offset);
  }
  function draw() {
    if (!ready) return;
    const anim = pack.animations[mode];
    const frame = Math.min(anim.frames.length - 1, Math.floor(elapsed / anim.duration_ms));
    $('timeline').value = frame; $('time-label').textContent = (frame+1) + ' / ' + anim.frames.length;
    $('frame-label').textContent = paused ? 'ПАУЗА' : mode.toUpperCase();
    ctx.clearRect(0,0,768,768); ctx.save();
    const eyes = custom ? $('eyes').value : pack.expressions[expression].eyes;
    const mouth = custom ? $('mouth').value : pack.expressions[expression].mouth;
    if (turn) ctx.drawImage(images.get(turn),0,0);
    else if (mode !== 'idle' && !custom && expression === 'neutral') ctx.drawImage(images.get(anim.frames[frame]),0,0);
    else if (mode === 'blink') face(frame >= 2 && frame <= 3 ? 'closed' : frame === 1 || frame === 4 ? 'half' : eyes, mouth);
    else if (mode === 'talk') face(eyes, ['neutral','talk_a','talk_e','open','talk_a','o','talk_e','neutral'][frame]);
    else {
      if (mode === 'nod') ctx.translate(0,Math.sin(frame / (anim.frames.length-1)*Math.PI*2)*14);
      if (mode === 'shake') ctx.translate(Math.sin(frame / (anim.frames.length-1)*Math.PI*4)*15,0);
      if (mode === 'idle') ctx.translate(0,Math.sin(frame / anim.frames.length*Math.PI*2)*4);
      if (custom) face(eyes,mouth); else ctx.drawImage(images.get(pack.expressions[expression].file),0,0);
    }
    ctx.restore();
  }
  function tick(now) {
    if (ready && !paused && !document.hidden) {
      const delta = previous === null ? 0 : Math.min(now - previous, 100) * motionSpeed(mode);
      elapsed += delta; quietTime += delta;
      const anim = pack.animations[mode], length = anim.frames.length * anim.duration_ms;
      if (elapsed >= length) {
        if (mode === 'idle' || mode === 'talk') elapsed %= length;
        else { mode = 'idle'; elapsed = 0; quietTime = 0; blinkAt = 3000 + Math.random()*3000; syncControls(); }
      }
      if (mode === 'idle' && !turn && $('auto-blink').checked && quietTime > blinkAt) {
        mode = 'blink'; elapsed = 0; quietTime = 0; syncControls();
      }
      draw();
    }
    previous = now;
    if (!document.hidden) raf = requestAnimationFrame(tick);
  }
  document.addEventListener('visibilitychange', () => {
    previous = null;
    if (raf !== null) cancelAnimationFrame(raf);
    if (document.hidden) stopAudio(); else raf = requestAnimationFrame(tick);
  });
  window.addEventListener('pagehide', stopAudio);
  $('reaction-sound').onchange = () => { if (!$('reaction-sound').checked) stopAudio(); };
  function selectedSound() { return sounds.find(x => x.id === $('sound').value); }
  function soundDescription() {
    const item = selectedSound();
    $('sound-description').textContent = item ? item.description_ru + ' · ' + item.duration_s.toFixed(2) + ' с' : '';
  }
  function soundOptions() {
    stopAudio(); $('sound').replaceChildren();
    const category = $('sound-category').value;
    for (const item of sounds.filter(x => x.category === category)) $('sound').add(new Option(item.description_ru + ' · ' + item.id.split('/').pop(),item.id));
    soundDescription();
  }
  $('sound-category').onchange = soundOptions;
  $('sound').onchange = () => { stopAudio(); soundDescription(); };
  $('play-sound').onclick = () => playSound(selectedSound());
  $('stop-sound').onclick = () => { stopAudio(); notice('Звук остановлен.'); };
  $('volume').oninput = () => { $('volume-label').textContent = $('volume').value + '%'; if (audio) audio.volume = Number($('volume').value)/100; };
  $('sound-count').textContent = sounds.length + ' звуков';
  soundOptions(); syncControls();
  const paths = new Set([pack.character_base,...pack.eye_parts,...pack.mouth_parts,...Object.values(pack.expressions).map(x=>x.file),...Object.values(pack.turns),...Object.values(pack.animations).flatMap(x=>x.frames)]);
  document.querySelectorAll('#motions button,#apply-face,#pause').forEach(b=>b.disabled=true);
  Promise.all([...paths].map(path => new Promise((resolve,reject) => {
    const img = new Image(); img.onload = () => { images.set(path,img); resolve(); };
    img.onerror = () => reject(new Error(path)); img.src = imageRoot + path;
  }))).then(() => {
    ready = true;
    document.querySelectorAll('#motions button,#apply-face,#pause').forEach(b=>b.disabled=false);
    $('catalog-status').textContent = paths.size + ' PNG · ' + Object.keys(pack.animations).length + ' анимаций · ' + data.sounds.files.length + ' звуков';
    draw(); raf = requestAnimationFrame(tick);
  }).catch(error => { $('catalog-status').textContent = 'Не удалось загрузить: ' + error.message; notice('Проверьте расположение исходных пакетов рядом со стендом.'); });
})();
