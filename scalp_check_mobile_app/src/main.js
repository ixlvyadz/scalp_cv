import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import './styles.css';
import { icon, logo, CHECK, WARN, HAIR_MARK } from './icons.js';
import { APP, CONDITIONS, conditionByKey, conditionForLabel, samplePhoto } from './content.js';
import { loadModel, predict, info } from './model.js';
import { checkQuality, lookedMap, drawOverlay } from './analysis.js';

const app = document.getElementById('app');
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const out100 = (p) => Math.round(p * 100);

// Last scan (kept in memory only; never saved).
const state = { photo: null, quality: null, result: null, overlay: null, busy: false, error: null };

// ---------- Shared pieces ----------
const NOTE = 'This is a guide only, not a medical diagnosis.';

function tabbar(active) {
  const tabs = [['Home', 'home', '#/home'], ['Scan', 'cam', '#/scan'], ['Learn', 'book', '#/learn'], ['About', 'info', '#/about']];
  return `<nav class="tabbar" aria-label="Main">${tabs.map(([n, i, h]) => {
    const on = n === active;
    return `<button class="tab" data-go="${h}"${on ? ' aria-current="page"' : ''}><span class="pillbg">${icon(i, 24, on ? '#0F6E7A' : '#4F6670')}</span>${n}</button>`;
  }).join('')}</nav>`;
}

function innerHeader(title, sub, back = 'back') {
  return `<header class="hdr hdr-inner">${HAIR_MARK}<div class="row">
    <button class="back" aria-label="Back" data-back="${back}">${icon('back', 22, '#fff')}</button>
    <div><h1 class="t" style="margin:0">${esc(title)}</h1><div class="s">${esc(sub)}</div></div></div></header>`;
}

function tabHeader(title, sub) {
  return `<header class="hdr hdr-tab">${HAIR_MARK}<div><h1 class="t" style="margin:0">${esc(title)}</h1><div class="s">${esc(sub)}</div></div></header>`;
}

function frame(height, inner = '', label = '') {
  return `<div class="frame" style="height:${height}px"${label ? ` role="img" aria-label="${esc(label)}"` : ''}>${inner}<div class="k k1"></div><div class="k k2"></div><div class="k k3"></div><div class="k k4"></div></div>`;
}

function page(body, tab) {
  app.innerHTML = `<main class="screen" id="scroll">${body}</main>${tabbar(tab)}`;
  app.querySelector('#scroll').scrollTop = 0;
}

// ---------- Screens ----------
function Home() {
  const chips = ['alopecia_areata', 'psoriasis', 'folliculitis', 'head_lice'].map((k) => conditionByKey(k));
  page(`
    <header class="hdr hdr-home">${HAIR_MARK}
      <div class="brand">${logo('#fff', '#0F6E7A', 30)}Scalp Check</div>
      <div class="hello">${greeting()}</div>
      <h1 class="home-title">How is your scalp today?</h1>
      <div class="priv">${icon('lock', 16, '#fff')}Private: photos are never saved</div>
    </header>
    <div class="home-body">
      <section class="card lift scan-card">
        <div class="top"><div class="tile48">${icon('cam', 26)}</div>
          <div><div class="t">Start a scan</div><div class="d">Take a close, well-lit photo of the area you are worried about.</div></div></div>
        <div style="margin-top:16px"><button class="btn" data-go="#/scan">Scan now</button></div>
      </section>
      <div class="sec-row"><h2 class="h17" style="margin:0">What we can look at</h2><button class="link" data-go="#/learn">See all 10</button></div>
      <div class="chips">${chips.map((c) => `<button class="chip" data-go="#/learn/${c.key}">${esc(c.name)}</button>`).join('')}<button class="chip" data-go="#/learn">+ 6 more</button></div>
      <h2 class="h17" style="margin:0">Tips for a good photo</h2>
      ${tipRow('sun', 'Use natural light')}${tipRow('scan', 'Get close to the area')}${tipRow('lock', 'Your photo is not saved')}
      <div class="small">${NOTE}</div>
    </div>`, 'Home');
}
const tipRow = (i, t, s = '') => `<div class="irow"><div class="circ40">${icon(i, 24)}</div><div><div class="t">${esc(t)}</div>${s ? `<div class="s">${esc(s)}</div>` : ''}</div></div>`;
function greeting() {
  const h = new Date().getHours();
  return h < 12 ? 'Good morning,' : h < 18 ? 'Good afternoon,' : 'Good evening,';
}

function qualityPills(q) {
  if (!q) return `<span class="pill tip">Clear</span><span class="pill tip">Bright</span><span class="pill tip">Close</span>`;
  const p = (ok, yes, no) => ok ? `<span class="pill">${CHECK}${yes}</span>` : `<span class="pill warn">${WARN}${no}</span>`;
  return p(q.clear, 'Clear', 'Blurry') + p(q.bright, 'Bright', 'Too dark') + `<span class="pill tip">Close</span>`;
}

function Scan() {
  const inner = state.photo
    ? `<img src="${state.photo.toDataURL('image/jpeg', 0.9)}" alt="Your photo">${state.busy ? `<div class="busy"><div class="spinner"></div>Looking at your photo…</div>` : ''}`
    : `<div class="ph">${icon('cam', 40, 'rgba(255,255,255,.85)')}</div>`;
  page(`
    ${innerHeader('New scan', 'Take a clear photo of the area', '#/home')}
    <div class="body">
      <div style="font-size:15px;color:var(--slate)">Place the affected area inside the frame.</div>
      ${frame(300, inner)}
      <div class="pills">${qualityPills(state.quality)}</div>
      ${state.error ? `<div class="err">${esc(state.error)}</div>` : ''}
      <button class="btn" id="take" ${state.busy ? 'disabled' : ''}>Take photo</button>
      <button class="btn alt" id="pick" ${state.busy ? 'disabled' : ''}>Choose from gallery</button>
      <div class="center" style="font-size:13px;color:var(--slate)">Part the hair and hold the phone steady.</div>
      <input type="file" id="fcam" accept="image/*" capture="environment" hidden>
      <input type="file" id="fgal" accept="image/*" hidden>
    </div>`, 'Scan');
  const cam = app.querySelector('#fcam'), gal = app.querySelector('#fgal');
  app.querySelector('#take').onclick = () => cam.click();
  app.querySelector('#pick').onclick = () => gal.click();
  cam.onchange = gal.onchange = (e) => e.target.files?.[0] && analyse(e.target.files[0]);
}

async function analyse(file) {
  state.error = null; state.overlay = null; state.result = null;
  state.busy = true;
  try {
    await loadModel();
    const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' }).catch(() => createImageBitmap(file));
    const res = await predict(bmp);
    bmp.close?.();
    state.photo = res.canvas;
    state.quality = checkQuality(res.canvas);
    Scan(); // show the photo with the "looking" overlay
    await new Promise((r) => setTimeout(r, 500));
    state.result = res;
    state.busy = false;
    location.hash = '#/result';
  } catch (e) {
    console.error(e);
    state.busy = false;
    state.error = 'Sorry, the app could not check this photo. Please try another photo. (' + (e.message || e) + ')';
    Scan();
  }
}

function Result() {
  const r = state.result;
  if (!r) return (location.hash = '#/scan');
  if (!r.sure) return NotSure();
  const best = conditionForLabel(r.top[0].label);
  const others = r.top.slice(1).map((t) => ({ c: conditionForLabel(t.label), p: t.p }));
  const pct = out100(r.top[0].p);
  page(`
    ${innerHeader('Your result', 'Here is what we found', '#/scan')}
    <div class="body g10">
      <section class="card">
        <span class="pill">${CHECK}We're fairly sure</span>
        <div class="lab" style="margin-top:12px">Most likely</div>
        <h2 class="res-name">${esc(best.name)}</h2>
        <div style="font-size:14px;color:var(--slate)">${esc(best.short)}</div>
        <div class="sure-row"><b>How sure are we?</b><span>${pct} out of 100</span></div>
        <div class="meter" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}" aria-label="How sure"><div style="width:${pct}%"></div></div>
        <div class="ends"><span>Not sure</span><span>Very sure</span></div>
      </section>
      ${best.what ? `<section class="card" style="display:flex;flex-direction:column;gap:10px">
        <div class="lab">About this condition</div>
        <div class="para">${esc(best.resultWhat || best.what)}</div>
        <div class="lab">Common signs</div>
        <div class="dot-list">${(best.resultSigns || best.signs.slice(0, 3)).map((s) => `<div class="dl">${esc(s)}</div>`).join('')}</div>
        ${best.good ? `<div class="good"><b>Good to know:</b> ${esc(best.good)} A skin doctor can confirm it and suggest the right care.</div>` : ''}
        ${best.source ? `<div class="tiny">${esc(best.source)}</div>` : ''}
      </section>` : ''}
      <section class="card"><div class="lab">Other possibilities</div>
        <div class="bars">${others.map(({ c, p }) => `<div><div class="bar-top"><span>${esc(c.name)}</span><span>${out100(p)} out of 100</span></div><div class="track"><div style="width:${out100(p)}%"></div></div></div>`).join('')}</div>
      </section>
      <section class="card"><div class="note">${icon('info', 22)}<span>Please show this result to a skin doctor for a proper check. ${NOTE}</span></div></section>
      <button class="btn" data-go="#/why">See why we think so</button>
      <button class="btn alt" data-go="#/scan" data-reset>Scan again</button>
    </div>`, 'Scan');
}

function NotSure() {
  const q = state.quality || {};
  const reasons = [];
  if (q.bright === false) reasons.push('The photo is too dark');
  if (q.clear === false) reasons.push('The area is blurry or too far away');
  if (!reasons.length) reasons.push('The photo is too dark or blurry', 'The area is too far away');
  reasons.push('It may be something we do not cover');
  page(`
    ${innerHeader('Your result', 'We could not tell this time', '#/scan')}
    <div class="body">
      <div class="ns-top"><div class="ns-ic">${icon('info', 36, '#7A5200')}</div>
        <h2>We are not sure</h2>
        <p>We cannot give a reliable answer from this photo. If you are worried, please see a skin doctor.</p></div>
      <section class="card"><div class="lab">Why this may have happened</div>
        <div class="reasons">${reasons.map((t) => `<div>${WARN}${esc(t)}</div>`).join('')}</div></section>
      <button class="btn" data-go="#/scan" data-reset data-open="camera">Retake photo</button>
      <button class="btn alt" data-go="#/scan" data-reset data-open="gallery">Choose another photo</button>
      <div class="center small">${NOTE}</div>
    </div>`, 'Scan');
}

async function Why() {
  const r = state.result;
  if (!r) return (location.hash = '#/scan');
  const ready = !!state.overlay;
  page(`
    ${innerHeader('Why we think so', 'See where the app looked', '#/result')}
    <div class="body g16">
      <div class="pair">
        <div><div class="cap">Your photo</div>${frame(190, `<img src="${state.photo.toDataURL('image/jpeg', 0.9)}" alt="Your photo">`)}</div>
        <div><div class="cap">Where the app looked</div>${frame(190, ready
          ? `<img src="${state.overlay.toDataURL('image/jpeg', 0.9)}" alt="Your photo with the areas the app relied on highlighted">`
          : `<div class="busy" id="lk"><div class="spinner"></div><span id="lkp">Working… 0%</span></div>`)}</div>
      </div>
      <div class="legend"><span>Looked less</span><div class="ramp"></div><span>Looked more</span></div>
      <section class="card"><div class="lab">How to read this</div>
        <div class="para" style="margin-top:8px">The bright areas show where the app looked most when it made its guess. Check that they sit on the affected skin, not on the background or your hair.</div>
        <div class="para" style="margin-top:8px;color:var(--slate)">This is a guide, not proof.</div></section>
      <button class="btn" data-go="#/result">Back to result</button>
    </div>`, 'Scan');
  if (ready) return;
  try {
    const top = r.top[0];
    const map = await lookedMap(r.data, top.index, top.p, (f) => {
      const el = document.getElementById('lkp');
      if (el) el.textContent = `Working… ${Math.round(f * 100)}%`;
    });
    state.overlay = drawOverlay(state.photo, map);
    if (location.hash === '#/why') Why();
  } catch (e) {
    const el = document.getElementById('lk');
    if (el) el.innerHTML = 'Could not make this view.';
  }
}

function Learn() {
  page(`
    ${tabHeader('Learn', 'Skin and hair conditions we can look at. Tap one to read more.')}
    <div class="body" style="gap:0">${CONDITIONS.map((c) => `
      <button class="lrow" data-go="#/learn/${c.key}">${thumb(c)}
        <div class="tx"><div class="n">${esc(c.name)}</div><div class="d">${esc(c.short)}</div></div>${icon('next', 20, '#4F6670')}</button>`).join('')}
    </div>`, 'Learn');
}
// Photo if public/samples/<key>.jpg exists, otherwise the placeholder colour shows through.
const thumb = (c) => `<div class="thumb" style="background:url('${samplePhoto(c.key)}') center/cover, var(--photo)"></div>`;

function Condition(key) {
  const c = conditionByKey(key);
  if (!c) return (location.hash = '#/learn');
  const look = c.lookalike && conditionByKey(c.lookalike.key);
  page(`
    ${innerHeader(c.name, c.subtitle, '#/learn')}
    <div class="body">
      ${frame(200, `<div class="ph">Sample photo</div><div style="position:absolute;inset:0;background:url('${samplePhoto(c.key)}') center/cover"></div>`, `Sample photo of ${c.name}`)}
      <div class="pills">${c.facts.map((f) => `<span class="pill">${CHECK.replace('width="16" height="16"', 'width="14" height="14"')}${esc(f)}</span>`).join('')}</div>
      <section class="card"><div class="lab">What it is</div><div class="para" style="margin-top:8px">${esc(c.what)}</div></section>
      <section class="card"><div class="lab">Common signs</div><div class="dot-list" style="margin-top:10px">${c.signs.map((s) => `<div class="dl">${esc(s)}</div>`).join('')}</div></section>
      ${look ? `<section class="card"><div class="lab">Can look like</div>
        <button class="lk" data-go="#/learn/${look.key}"><div><div class="n">${esc(look.name)}</div><div class="d">${esc(c.lookalike.text)}</div></div>${icon('next', 20, '#4F6670')}</button></section>` : ''}
      <section class="card"><div class="lab">When to see a skin doctor</div><div class="dot-list" style="margin-top:10px">${c.doctor.map((s) => `<div class="dl amber">${esc(s)}</div>`).join('')}</div></section>
      ${c.good ? `<div class="good"><b>Good to know:</b> ${esc(c.good)}</div>` : ''}
      <button class="btn" data-go="#/scan" data-reset>Scan my scalp</button>
      <button class="btn alt" data-go="#/learn">Back to all conditions</button>
      <div class="tiny">${esc(c.source || 'General information for learning.')} This is a guide, not a diagnosis.</div>
    </div>`, 'Learn');
}

const FAQ = () => [
  ['Is this a diagnosis?', 'No. Scalp Check is a guide for learning and for deciding when to see a doctor. Only a skin doctor can diagnose.'],
  ['How accurate is it?', APP.accuracyOutOf100 == null
    ? 'We are still testing. When testing is finished, this answer will say how many photos out of 100 it names correctly: <b>— out of 100</b>. We test it on photos it has never seen before.'
    : `In our tests, it named the condition correctly in about <b>${APP.accuracyOutOf100} out of 100</b> photos. We tested it on photos it had never seen before.`],
  ['How does Scalp Check work?', 'You take a close photo of the area. The app compares it with thousands of example photos it learned from, and tells you which of ten conditions it looks most like.'],
  ['What happens to my photo?', 'Your photo is used only to give you a result, right here on your phone. It is not saved or sent anywhere.'],
];

function About() {
  page(`
    ${tabHeader('About', 'Scalp Check, your privacy and the team')}
    <div class="body">
      <section class="card"><div class="lab">What is Scalp Check?</div>
        <div style="font-size:14px;margin-top:8px;line-height:1.45">A simple app that looks at a photo of your scalp or hair and tells you which of ten common conditions it most looks like. It also shows where it looked, and tells you honestly when it is not sure.</div></section>
      <section class="card">${tipRow('lock', 'Your privacy', 'Your photos are never saved.')}</section>
      <section class="card"><div class="lab">Meet the team</div>
        <div class="team">${APP.team.map((m) => `<div class="prof"><div class="av">${icon('user', 26)}</div><div class="n">${esc(m.name)}</div><div class="small">${esc(m.role)}</div></div>`).join('')}</div>
        <div class="small" style="margin-top:8px">${esc(APP.course)}</div></section>
      <section class="card"><div class="lab">Common questions</div>
        <div class="faq">${FAQ().map(([q, a], i) => `
          <button class="faq-q" aria-expanded="false" aria-controls="fa${i}"><span>${q}</span>${icon('down', 20, '#4F6670')}</button>
          <div class="faq-a" id="fa${i}" hidden>${a}</div>`).join('')}</div></section>
      <section class="card p8"><div class="amber-box">Scalp Check is a school project and a guide only. It cannot diagnose. Please see a dermatologist for any concern.</div></section>
    </div>`, 'About');
  app.querySelectorAll('.faq-q').forEach((b) => (b.onclick = () => {
    const open = b.getAttribute('aria-expanded') === 'true';
    b.setAttribute('aria-expanded', String(!open));
    document.getElementById(b.getAttribute('aria-controls')).hidden = open;
  }));
}

// Hidden check screen for the team (#/check): shows what the app read from the model files.
async function Check() {
  page(`${innerHeader('Model check', 'For the team only', '#/about')}<div class="body"><div class="card" id="ck">Loading…</div></div>`, 'About');
  try {
    await loadModel();
    document.getElementById('ck').innerHTML = `<pre style="white-space:pre-wrap;font-size:12px;margin:0">${esc(JSON.stringify({
      modelFile: info.modelFile, layout: info.layout, preprocess: info.pre, temperature: info.temperature,
      threshold: info.threshold, labels: info.labels, matchedTo: info.labels.map((l) => conditionForLabel(l).name), warnings: info.warnings,
    }, null, 2))}</pre>`;
  } catch (e) {
    document.getElementById('ck').innerHTML = `<div class="err">${esc(e.message || e)}</div>`;
  }
}

// ---------- Router ----------
function route() {
  const h = location.hash || '#/home';
  const [, a, b] = h.split('/');
  switch (a) {
    case 'scan': return Scan();
    case 'result': return Result();
    case 'why': return Why();
    case 'learn': return b ? Condition(b) : Learn();
    case 'about': return About();
    case 'check': return Check();
    default: return Home();
  }
}

app.addEventListener('click', (e) => {
  const el = e.target.closest('[data-go],[data-back]');
  if (!el) return;
  if (el.hasAttribute('data-reset')) { state.photo = null; state.quality = null; state.result = null; state.overlay = null; state.error = null; }
  if (el.hasAttribute('data-back')) {
    location.hash = el.dataset.back;
    return;
  }
  const open = el.dataset.open;
  if (location.hash === el.dataset.go) route(); else location.hash = el.dataset.go;
  if (open) setTimeout(() => app.querySelector(open === 'camera' ? '#fcam' : '#fgal')?.click(), 50);
});
window.addEventListener('hashchange', route);
route();
loadModel().catch((e) => console.warn('Model not loaded yet:', e.message));
