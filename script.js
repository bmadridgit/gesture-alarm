import { GestureRecognizer, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

/* ---------- config ---------- */
const HOLD_MS = 5000, MIN_SCORE = 0.7, SEQ_LEN = 3;

/* ---------- hand outlines (200x250 viewBox) ----------
   s = silhouette shapes (only the outer edge is drawn), l = inner detail lines (finger gaps, joints) */
const cap = (x1,y1,x2,y2,w=17) => {
  const dx=x2-x1, dy=y2-y1, L=Math.hypot(dx,dy), a=Math.atan2(dy,dx)*180/Math.PI;
  return `<rect x="${-L/2-w/2}" y="${-w/2}" width="${L+w}" height="${w}" rx="${w/2}" transform="translate(${(x1+x2)/2} ${(y1+y2)/2}) rotate(${a})"/>`;
};
const raw = (s, l='') => ({ s, l });
// extended finger: capsule plus two joint creases
const fin = (x1,y1,x2,y2,w=17) => {
  const dx=x2-x1, dy=y2-y1, L=Math.hypot(dx,dy), nx=-dy/L, ny=dx/L, h=w*.28;
  const ticks = [.42,.7].map(f => { const px=x1+dx*f, py=y1+dy*f;
    return `M${(px-nx*h).toFixed(1)} ${(py-ny*h).toFixed(1)}L${(px+nx*h).toFixed(1)} ${(py+ny*h).toFixed(1)}`; }).join('');
  return { s: cap(x1,y1,x2,y2,w), l: ticks };
};
const mk = (...p) => ({ s: p.map(x=>x.s).join(''), l: p.map(x=>x.l||'').join('') });

const OPEN_PALM = raw('<rect x="57" y="118" width="86" height="76" rx="30"/>');
// fist seen from the front: four curled finger pads, thumb folded across
const FIST_BODY = raw('<rect x="55" y="106" width="90" height="86" rx="34"/>');
const pad = x => raw(cap(x,114,x,148,20));
const thumbAcross = (y1,y2,x2=120) => raw(cap(72,y1,x2,y2,19),
  `M70.4 ${y1-9.4}L${x2-1.6} ${y2-9.4}`);
// fist seen from the side (thumbs up/down): stacked curled fingers
const SIDE_BODY = raw('<rect x="60" y="120" width="70" height="80" rx="28"/>',
  'M84 140.5H128M84 157.5H128M84 174.5H128');
const bands = [132,149,166,183].map(y => raw(cap(82,y,128,y)));

const GESTURES = {
  Open_Palm: { label:"Open palm", art: mk(OPEN_PALM,
    fin(70,130,64,60), fin(89,128,88,50), fin(108,130,112,58), fin(127,134,136,76),
    fin(60,170,38,146,18), fin(38,146,28,120,16)) },
  Closed_Fist: { label:"Fist", art: mk(FIST_BODY, pad(68), pad(89), pad(110), pad(131),
    raw('', 'M78.5 120V156M99.5 120V156M120.5 120V156'), thumbAcross(166,158)) },
  Thumb_Up: { label:"Thumbs up", art: mk(SIDE_BODY, ...bands,
    fin(76,126,74,88,19), fin(74,88,74,58,17)) },
  Thumb_Down: { label:"Thumbs down", flip:true, art: mk(SIDE_BODY, ...bands,
    fin(76,126,74,88,19), fin(74,88,74,58,17)) },
  Pointing_Up: { label:"Point up", art: mk(FIST_BODY, pad(89), pad(110), pad(131),
    fin(68,124,66,52,18), raw('', 'M78.5 130V156M99.5 120V156M120.5 120V156'),
    thumbAcross(166,158)) },
  Victory: { label:"Peace sign", art: mk(FIST_BODY, pad(110), pad(131),
    fin(70,126,56,54), fin(92,126,104,46), raw('', 'M120.5 120V150'),
    thumbAcross(168,160,124)) },
  Okay: { label:"Okay", art: mk(OPEN_PALM, raw('<circle cx="80" cy="108" r="19"/>'),
    fin(64,168,66,132,19), fin(104,128,108,52), fin(121,130,128,60), fin(136,138,148,86),
    raw('', 'M71 108a9 9 0 1 0 18 0a9 9 0 1 0-18 0')) }
};

let uid = 0;
function handSVG(g) {
  const id = "m" + (uid++);
  const wrap = x => g.flip ? `<g transform="translate(0 250) scale(1 -1)">${x}</g>` : x;
  // mask hides the stroke halves inside the hand, so only the outer silhouette shows
  return `<svg viewBox="0 0 200 250" aria-hidden="true">
    <mask id="${id}"><rect width="200" height="250" fill="#fff"/><g fill="#000">${wrap(g.art.s)}</g></mask>
    <g class="fill">${wrap(g.art.s)}</g>
    <g class="outline" mask="url(#${id})">${wrap(g.art.s)}</g>
    ${wrap(`<path class="detail" d="${g.art.l}"/>`)}</svg>`;
}

/* ---------- okay detection (not a MediaPipe preset) ---------- */
const d = (a,b) => Math.hypot(a.x-b.x, a.y-b.y);
function isOkay(lm) {
  const palm = d(lm[0], lm[9]);
  const loop = d(lm[4], lm[8]) < 0.35 * palm;
  const up = [[12,10],[16,14],[20,18]].every(([t,p]) => d(lm[t],lm[0]) > d(lm[p],lm[0]));
  return loop && up;
}
// is the hand roughly inside the on-screen outline? (landmarks are 0–1 of the video frame)
function inOutline(lm) {
  const xs = lm.map(p=>p.x), ys = lm.map(p=>p.y);
  const w = Math.max(...xs)-Math.min(...xs), h = Math.max(...ys)-Math.min(...ys);
  const cx = (Math.max(...xs)+Math.min(...xs))/2, cy = (Math.max(...ys)+Math.min(...ys))/2;
  return cx>.3 && cx<.7 && cy>.3 && cy<.75 && w>.15 && w<.6 && h>.2 && h<.75;
}

/* ---------- model ---------- */
let recognizer;
async function loadModel() {
  const fs = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  recognizer = await GestureRecognizer.createFromOptions(fs, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task" },
    runningMode: "VIDEO", numHands: 1
  });
}

/* ---------- sound: soft marimba-style chime that fades in ---------- */
let ctx, master, loopTimer;
const MOTIF = [659.25, 783.99, 987.77, 783.99, 659.25, 783.99, 987.77, 1318.51]; // E5 G5 B5 ...
const NOTE_GAP = 0.16, REPEAT_MS = 2600, FADE_IN_S = 25;   // tweak these to taste

function note(freq, t, dur = 0.55) {
  const g = ctx.createGain(), o1 = ctx.createOscillator(), o2 = ctx.createOscillator(), g2 = ctx.createGain();
  o1.type = o2.type = "sine";
  o1.frequency.value = freq; o2.frequency.value = freq * 4;   // quiet overtone gives the wooden "tock"
  g2.gain.value = 0.12;
  g.gain.setValueAtTime(0, t);
  g.gain.linearRampToValueAtTime(0.5, t + 0.006);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  o1.connect(g); o2.connect(g2).connect(g); g.connect(master);
  o1.start(t); o2.start(t); o1.stop(t + dur); o2.stop(t + dur);
}
function startSound() {
  master = ctx.createGain();
  master.gain.setValueAtTime(0.15, ctx.currentTime);
  master.gain.linearRampToValueAtTime(1, ctx.currentTime + FADE_IN_S);  // starts gentle, gets louder
  master.connect(ctx.destination);
  const play = () => { const t = ctx.currentTime + 0.05; MOTIF.forEach((f,i) => note(f, t + i * NOTE_GAP)); };
  play(); loopTimer = setInterval(play, REPEAT_MS);
}
function stopSound() {
  clearInterval(loopTimer);
  if (master) { master.gain.cancelScheduledValues(ctx.currentTime); master.gain.setTargetAtTime(0, ctx.currentTime, 0.05); }
}

/* ---------- UI helpers ---------- */
const $ = id => document.getElementById(id);
const show = id => document.querySelectorAll(".screen").forEach(s => s.classList.toggle("active", s.id === id));
function setStatus(text, go=false) {
  $("status").classList.toggle("go", go);
  $("status").querySelector("span").textContent = text;
}
function setProgress(step) {
  [...$("segs").children].forEach((s,i) => {
    s.classList.toggle("on", i <= step);
    s.classList.toggle("cur", i === step);
  });
}
function showTarget(key) {
  $("hand").innerHTML = handSVG(GESTURES[key]);
  $("name").textContent = GESTURES[key].label;
  $("card").classList.remove("hit");
  $("ringc").style.strokeDashoffset = 100;
}

/* ---------- alarm flow ---------- */
let snoozing = false, stream, raf, seq, step, holdStart, cooldownUntil, ringStart;
const log = [];

function pickSequence() {
  const keys = Object.keys(GESTURES);
  for (let i = keys.length - 1; i > 0; i--) { const j = Math.floor(Math.random()*(i+1)); [keys[i],keys[j]] = [keys[j],keys[i]]; }
  return keys.slice(0, SEQ_LEN);   // distinct, random order
}

async function ring() {
  seq = pickSequence().slice(0, snoozing ? 1 : SEQ_LEN); step = 0;
  [...$("segs").children].forEach((b,i) => b.style.display = i < seq.length ? "" : "none");
  $("loading").textContent = "";
  holdStart = null; cooldownUntil = 0;
  log.length = 0; ringStart = performance.now();
  show("s-ring"); setProgress(0); showTarget(seq[0]); setStatus("Looking for hand…");
  startSound();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    $("video").srcObject = stream; await $("video").play();
    loop();
  } catch { setStatus("Camera blocked. Allow access to dismiss."); }
}

function loop() {
  const v = $("video"), now = performance.now();
  if (v.readyState >= 2 && now > cooldownUntil) {
    const res = recognizer.recognizeForVideo(v, now);
    const lm = res.landmarks[0];
    const top = res.gestures[0]?.[0];
    const target = seq[step];

    let seen = null;
    if (lm) seen = isOkay(lm) ? "Okay" : (top?.score >= MIN_SCORE ? top.categoryName : null);

    if (!lm) { reset("Looking for hand…"); }
    else if (!inOutline(lm)) { reset("Move your hand into the outline"); }
    else if (seen === target) {
      holdStart ??= now;
      const pct = Math.min(100, (now - holdStart) / HOLD_MS * 100);
      $("card").classList.add("hit");
      $("ringc").style.strokeDashoffset = 100 - pct;
      setStatus("Hold it…", true);
      if (pct >= 100) return advance();
    } else {
      reset(seen && seen !== "None" ? `That's not it. Try ${GESTURES[target].label.toLowerCase()}` : "Looking for hand…");
    }
  }
  raf = requestAnimationFrame(loop);
}

function reset(msg) {
  holdStart = null;
  $("card").classList.remove("hit");
  $("ringc").style.strokeDashoffset = 100;
  setStatus(msg);
}

function advance() {
  log.push({ gesture: seq[step], ms: Math.round(performance.now() - ringStart) });
  step++;
  if (step >= seq.length) return dismiss();
  setProgress(step); showTarget(seq[step]);
  holdStart = null; cooldownUntil = performance.now() + 700;   // lower your hand between gestures
  setStatus("Got it. Next gesture");
  raf = requestAnimationFrame(loop);
}

function dismiss() {
  cancelAnimationFrame(raf); stopSound();
  stream.getTracks().forEach(t => t.stop());
  console.table(log);
  if (snoozing) {                       // snooze: gesture done, ring again in 5 min (10 s for the demo)
    snoozing = false; show("s-start");
    $("loading").textContent = "Snoozed. Rings again in 10 seconds.";
    setTimeout(ring, 10000); return;
  }
  show("s-done");
}

/* ---------- wiring ---------- */
$("start").onclick = async () => {
  ctx = new AudioContext(); await ctx.resume();     // audio must start from a click
  $("loading").textContent = "Loading hand model…";
  if (!recognizer) await loadModel();
  $("loading").textContent = "";
  ring();
};
$("done").onclick = () => show("s-start");
$("snooze").onclick = () => { snoozing = true; ring(); };