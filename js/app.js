import { listFolders, getFolder, saveFolder, deleteFolder, createFolder, uid } from './storage.js';
import { toWorkingCanvas, segmentCharacters } from './segment.js';
import { getWorker, recognizeAll } from './ocr.js';
import { LETTERS, buildFont, fontToBlob } from './font.js';

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
};
const glyphImg = (src, alt = '') => el('img', { class: 'glyph', src, alt });

let current = null; // 目前開啟的資料夾
let stream = null;
let busy = false;

/* =========================================================
 * 對話框 & 提示
 * ======================================================= */
let modalResolve = null;

function showModal({ title, body, actions = [], dismissible = true }) {
  closeModal(undefined, true);
  const modal = $('#modal');
  $('#modal-title').textContent = title;
  const bodyEl = $('#modal-body');
  bodyEl.innerHTML = '';
  if (typeof body === 'string') bodyEl.append(el('p', {}, body));
  else if (body) bodyEl.append(body);
  const actionsEl = $('#modal-actions');
  actionsEl.innerHTML = '';
  for (const a of actions) {
    actionsEl.append(
      el('button', { class: `btn ${a.class || ''}`, onclick: () => closeModal(a.value) }, a.label)
    );
  }
  actionsEl.hidden = actions.length === 0;
  modal.hidden = false;
  modal.querySelector('.modal-backdrop').onclick = dismissible ? () => closeModal(null) : null;
  return new Promise((resolve) => (modalResolve = resolve));
}

function closeModal(value, silent = false) {
  const modal = $('#modal');
  if (modal.hidden && !modalResolve) return;
  modal.hidden = true;
  const r = modalResolve;
  modalResolve = null;
  if (r && !silent) r(value);
  else if (r) r(null);
}

let toastTimer = null;
function toast(msg, ms = 2600) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), ms);
}

async function promptText(title, placeholder = '', initial = '') {
  const input = el('input', { type: 'text', placeholder, value: initial, maxlength: 40 });
  const form = el('form', { onsubmit: (e) => { e.preventDefault(); closeModal('ok'); } }, input);
  const p = showModal({
    title,
    body: form,
    actions: [
      { label: '取消', value: null },
      { label: '確定', value: 'ok', class: 'btn-primary' },
    ],
  });
  setTimeout(() => input.focus(), 50);
  const r = await p;
  if (r !== 'ok') return null;
  return input.value.trim();
}

/* =========================================================
 * 路由
 * ======================================================= */
function route() {
  const m = location.hash.match(/^#\/folder\/(.+)$/);
  if (m) openFolder(decodeURIComponent(m[1]));
  else showHome();
}
window.addEventListener('hashchange', route);

/* =========================================================
 * 首頁
 * ======================================================= */
async function showHome() {
  stopCamera();
  current = null;
  $('#view-folder').hidden = true;
  $('#view-home').hidden = false;
  const folders = await listFolders();
  const list = $('#folder-list');
  list.innerHTML = '';
  $('#home-empty').hidden = folders.length > 0;
  for (const f of folders) {
    const done = LETTERS.filter((L) => f.glyphs[L]);
    const thumb = el('div', { class: 'folder-thumb' });
    ['A', 'B', 'C', 'D'].forEach((L) => {
      thumb.append(f.glyphs[L] ? glyphImg(f.glyphs[L], L) : el('span', {}, L));
    });
    const more = el(
      'button',
      {
        class: 'icon-btn folder-more',
        'aria-label': '更多選項',
        onclick: (e) => { e.stopPropagation(); folderMenu(f); },
      },
      el('span', { style: 'font-size:22px;line-height:1' }, '⋯')
    );
    list.append(
      el(
        'li',
        { class: 'folder-card', onclick: () => (location.hash = `#/folder/${encodeURIComponent(f.id)}`) },
        thumb,
        el(
          'div',
          { class: 'folder-info' },
          el('h3', {}, f.name),
          el('p', {}, done.length === 26 ? '26 / 26 · 可以輸出字型了！' : `${done.length} / 26 個字母`),
          el('div', { class: 'bar' }, el('i', { style: `width:${(done.length / 26) * 100}%` }))
        ),
        more
      )
    );
  }
}

async function folderMenu(f) {
  const action = await showModal({
    title: f.name,
    body: '要對這個資料夾做什麼呢？',
    actions: [
      { label: '重新命名', value: 'rename' },
      { label: '刪除', value: 'delete', class: 'btn-danger' },
    ],
  });
  if (action === 'rename') {
    const name = await promptText('重新命名資料夾', '資料夾名稱', f.name);
    if (name) {
      f.name = name;
      await saveFolder(f);
      showHome();
    }
  } else if (action === 'delete') {
    const ok = await showModal({
      title: '確定要刪除嗎？',
      body: `「${f.name}」裡的所有字形都會被刪除，無法復原。`,
      actions: [
        { label: '取消', value: false },
        { label: '刪除', value: true, class: 'btn-primary' },
      ],
    });
    if (ok) {
      await deleteFolder(f.id);
      showHome();
    }
  }
}

$('#btn-new-folder').addEventListener('click', async () => {
  const name = await promptText('新增資料夾', '例如：我的手寫字');
  if (name === null) return;
  const folder = await createFolder(name || '未命名字型');
  location.hash = `#/folder/${encodeURIComponent(folder.id)}`;
});

/* =========================================================
 * 資料夾頁
 * ======================================================= */
async function openFolder(id) {
  const folder = await getFolder(id);
  if (!folder) {
    toast('找不到這個資料夾');
    location.hash = '#/';
    return;
  }
  current = folder;
  $('#view-home').hidden = true;
  $('#view-folder').hidden = false;
  window.scrollTo(0, 0);
  renderFolder();
  startCamera();
  // 先在背景載入 OCR，拍照時就不用等太久
  getWorker().catch(() => {});
}

function renderFolder(flashLetters = []) {
  if (!current) return;
  $('#folder-name').textContent = current.name;
  const done = LETTERS.filter((L) => current.glyphs[L]).length;
  $('#folder-progress').textContent = `已建立 ${done} / 26 個字母`;

  const grid = $('#glyph-grid');
  grid.innerHTML = '';
  for (const L of LETTERS) {
    const src = current.glyphs[L];
    const cell = el(
      'button',
      {
        class: `glyph-cell${src ? ' done' : ''}${flashLetters.includes(L) ? ' flash' : ''}`,
        'aria-label': src ? `${L}：已建立` : `${L}：尚未建立`,
        'data-letter': L,
        onclick: () => glyphMenu(L),
      },
      el('span', { class: 'tag' }, L),
      src ? glyphImg(src, L) : el('span', { class: 'ghost' }, L)
    );
    grid.append(cell);
  }

  const pending = current.pending || [];
  $('#pending-section').hidden = pending.length === 0;
  $('#pending-count').textContent = `(${pending.length})`;
  const plist = $('#pending-list');
  plist.innerHTML = '';
  for (const p of pending) {
    plist.append(
      el('button', { class: 'pending-item', onclick: () => pendingMenu(p) }, glyphImg(p.image, '未辨識'))
    );
  }
}

/** 點擊字母格 */
async function glyphMenu(L) {
  const src = current.glyphs[L];
  if (!src) {
    await showModal({
      title: `「${L}」尚未建立`,
      body: `請在紙上寫一個大寫的 ${L}，然後用上方的相機拍下來。`,
      actions: [{ label: '好', value: true, class: 'btn-primary' }],
    });
    return;
  }
  const action = await showModal({
    title: `字母「${L}」`,
    body: el('div', { class: 'big-glyph' }, glyphImg(src, L)),
    actions: [
      { label: '刪除', value: 'delete', class: 'btn-danger' },
      { label: '改成其他字母', value: 'move' },
      { label: '關閉', value: null, class: 'btn-primary' },
    ],
  });
  if (action === 'delete') {
    delete current.glyphs[L];
    await saveFolder(current);
    renderFolder();
  } else if (action === 'move') {
    const target = await pickLetter(`這個字其實是…`, L);
    if (!target || target === L) return;
    delete current.glyphs[L];
    await assignWithConflict(target, [src]);
    await saveFolder(current);
    renderFolder([target]);
  }
}

/** 點擊未辨識的片段 */
async function pendingMenu(p) {
  const picker = letterPickerEl();
  const body = el(
    'div',
    {},
    el('div', { class: 'big-glyph', style: 'height:150px;margin-bottom:14px' }, glyphImg(p.image)),
    el('p', {}, '這個字是哪個字母？'),
    picker
  );
  const action = await showModal({
    title: '指定字母',
    body,
    actions: [
      { label: '丟掉這張', value: 'discard', class: 'btn-danger' },
      { label: '取消', value: null },
    ],
  });
  if (action === 'discard' || (action && action.letter)) {
    current.pending = current.pending.filter((x) => x.id !== p.id);
    if (action.letter) await assignWithConflict(action.letter, [p.image]);
    await saveFolder(current);
    renderFolder(action.letter ? [action.letter] : []);
  }
}

function letterPickerEl(exclude) {
  const wrap = el('div', { class: 'letter-picker' });
  for (const L of LETTERS) {
    if (L === exclude) continue;
    wrap.append(
      el(
        'button',
        { class: current.glyphs[L] ? 'filled' : '', onclick: () => closeModal({ letter: L }) },
        L
      )
    );
  }
  return wrap;
}

async function pickLetter(title, exclude) {
  const r = await showModal({
    title,
    body: el('div', {}, el('p', {}, '綠色代表該字母已經建立過。'), letterPickerEl(exclude)),
    actions: [{ label: '取消', value: null }],
  });
  return r && r.letter;
}

$('#btn-clear-pending').addEventListener('click', async () => {
  const ok = await showModal({
    title: '清除未辨識的字？',
    body: '這些切割出來但沒有辨識成功的圖片都會被丟掉。',
    actions: [
      { label: '取消', value: false },
      { label: '清除', value: true, class: 'btn-primary' },
    ],
  });
  if (ok) {
    current.pending = [];
    await saveFolder(current);
    renderFolder();
  }
});

$('#btn-back').addEventListener('click', () => (location.hash = '#/'));

/* =========================================================
 * 相機
 * ======================================================= */
async function startCamera() {
  const video = $('#video');
  const msg = $('#camera-msg');
  showSnapshot(false);
  if (stream) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    msg.hidden = false;
    msg.innerHTML = '這個瀏覽器無法使用相機<br>（需要 HTTPS），請改用「從相簿選擇」';
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1920 },
        height: { ideal: 1440 },
      },
    });
    if (!current) { stopCamera(); return; } // 使用者已離開
    video.srcObject = stream;
    await video.play().catch(() => {});
    msg.hidden = true;
  } catch (err) {
    stream = null;
    msg.hidden = false;
    msg.innerHTML = '';
    msg.append(
      el(
        'div',
        {},
        el('p', {}, '無法開啟相機，請允許相機權限，或改用「從相簿選擇」。'),
        el('button', { class: 'btn btn-sm', onclick: () => startCamera() }, '再試一次')
      )
    );
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  const video = $('#video');
  if (video) video.srcObject = null;
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopCamera();
  else if (current && !busy) startCamera();
});

function showSnapshot(on) {
  $('#snapshot').hidden = !on;
  $('#video').style.visibility = on ? 'hidden' : 'visible';
  $('#camera-guide').hidden = on;
}

function setStatus(text) {
  const s = $('#camera-status');
  s.hidden = !text;
  s.textContent = text || '';
}

$('#btn-shutter').addEventListener('click', async () => {
  const video = $('#video');
  if (!stream || !video.videoWidth) {
    toast('相機還沒準備好，也可以用「從相簿選擇」');
    return;
  }
  const work = toWorkingCanvas(video, video.videoWidth, video.videoHeight);
  await processPhoto(work);
});

$('#file-input').addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    const { source, width, height } = await loadImageFile(file);
    const work = toWorkingCanvas(source, width, height);
    await processPhoto(work);
  } catch (err) {
    console.error(err);
    toast('無法讀取這張圖片');
  }
});

async function loadImageFile(file) {
  if ('createImageBitmap' in window) {
    try {
      const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
      return { source: bmp, width: bmp.width, height: bmp.height };
    } catch (_) { /* 舊瀏覽器退回 <img> */ }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    return { source: img, width: img.naturalWidth, height: img.naturalHeight };
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

/* =========================================================
 * 拍照後處理：切割 → 辨識 → 處理重複 → 存檔
 * ======================================================= */
async function processPhoto(work) {
  if (busy || !current) return;
  busy = true;
  $('#btn-shutter').disabled = true;
  const folder = current;

  // 定格畫面，並在上面畫出偵測框
  const snap = $('#snapshot');
  snap.width = work.width;
  snap.height = work.height;
  const sctx = snap.getContext('2d');
  sctx.drawImage(work, 0, 0);
  showSnapshot(true);

  try {
    setStatus('正在切割文字…');
    await nextFrame();
    const pieces = segmentCharacters(work);
    if (!pieces.length) {
      await showModal({
        title: '沒有找到字',
        body: '照片裡沒有偵測到手寫字。請在光線充足的地方，用深色筆在白紙上寫字再拍一次。',
        actions: [{ label: '好', value: true, class: 'btn-primary' }],
      });
      return;
    }
    const lw = Math.max(2, Math.round(work.width / 300));
    sctx.lineWidth = lw;
    sctx.strokeStyle = 'rgba(255, 200, 0, 0.9)';
    for (const p of pieces) sctx.strokeRect(p.box.x, p.box.y, p.box.w, p.box.h);

    await getWorker((t) => setStatus(t));
    setStatus(`辨識中… 0 / ${pieces.length}`);
    const letters = await recognizeAll(
      pieces.map((p) => p.raw),
      (done, total, i, letter) => {
        setStatus(done < total ? `辨識中… ${done} / ${total}` : '再仔細看一次沒辨識出來的字…');
        drawLabel(sctx, pieces[i].box, letter, lw);
      }
    );
    const results = pieces.map((p, i) => ({ letter: letters[i], image: p.glyph.toDataURL('image/png') }));
    setStatus('');

    if (current !== folder) return; // 使用者已離開這個資料夾
    await applyResults(results);
  } catch (err) {
    console.error(err);
    await showModal({
      title: '處理失敗',
      body: String(err.message || err),
      actions: [{ label: '好', value: true, class: 'btn-primary' }],
    });
  } finally {
    setStatus('');
    busy = false;
    $('#btn-shutter').disabled = false;
    showSnapshot(false);
    if (current && !stream) startCamera();
  }
}

function drawLabel(ctx, box, letter, lw) {
  ctx.strokeStyle = letter ? 'rgba(60, 200, 120, 0.95)' : 'rgba(255, 90, 60, 0.95)';
  ctx.strokeRect(box.x, box.y, box.w, box.h);
  const size = Math.max(16, Math.round(lw * 9));
  ctx.font = `bold ${size}px sans-serif`;
  const text = letter || '?';
  const tw = ctx.measureText(text).width + size * 0.5;
  ctx.fillStyle = letter ? 'rgba(40, 160, 90, 0.95)' : 'rgba(220, 70, 40, 0.95)';
  const y = Math.max(0, box.y - size * 1.2);
  ctx.fillRect(box.x, y, tw, size * 1.2);
  ctx.fillStyle = '#fff';
  ctx.fillText(text, box.x + size * 0.25, y + size * 0.95);
}

const nextFrame = () => new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));

async function applyResults(results) {
  const byLetter = new Map();
  const unknown = [];
  for (const r of results) {
    if (!r.letter) unknown.push(r.image);
    else {
      if (!byLetter.has(r.letter)) byLetter.set(r.letter, []);
      byLetter.get(r.letter).push(r.image);
    }
  }

  const added = [];
  const dupLetters = [...byLetter.keys()].filter((L) => byLetter.get(L).length > 1);
  if (dupLetters.length) {
    await showModal({
      title: '有重複的字喔！',
      body: el(
        'div',
        {},
        el('p', {}, `這張照片裡有重複的字母：${dupLetters.join('、')}。`),
        el('p', {}, '接下來請點選你最喜歡的那一個保留。')
      ),
      actions: [{ label: '開始挑選', value: true, class: 'btn-primary' }],
      dismissible: false,
    });
  }

  let keepExistingRest = false;
  for (const L of LETTERS) {
    const images = byLetter.get(L);
    if (!images) continue;
    // 使用者選了「其餘都保留原本的」：只拍到一次、且原本已建立的字就跳過
    if (keepExistingRest && images.length === 1 && current.glyphs[L]) continue;
    const changed = await assignWithConflict(L, images, { offerKeepAll: true });
    if (changed === 'keep-all') keepExistingRest = true;
    else if (changed) added.push(L);
  }

  if (unknown.length) {
    current.pending = (current.pending || []).concat(unknown.map((image) => ({ id: uid(), image })));
  }
  await saveFolder(current);
  renderFolder(added);

  const parts = [];
  if (added.length) parts.push(`新增 / 更新了 ${added.length} 個字母：${added.join(' ')}`);
  if (unknown.length) parts.push(`${unknown.length} 個字沒辨識出來，可以在下方手動指定`);
  toast(parts.join('；') || '沒有新增任何字母', 4000);
}

/**
 * 把候選圖片放到字母 L。若候選不只一個（同一張照片重複、或該字母已經有字形），
 * 就讓使用者挑選要保留哪一個。回傳是否有變更。
 */
async function assignWithConflict(L, images, { offerKeepAll = false } = {}) {
  const existing = current.glyphs[L];
  if (images.length === 1 && !existing) {
    current.glyphs[L] = images[0];
    return true;
  }
  const choices = images.map((src) => ({ src, isExisting: false }));
  if (existing) choices.unshift({ src: existing, isExisting: true });

  const grid = el('div', { class: 'choices' });
  choices.forEach((c, i) => {
    grid.append(
      el(
        'button',
        { class: 'choice', 'aria-label': `選擇第 ${i + 1} 個`, onclick: () => closeModal({ pick: i }) },
        c.isExisting ? el('span', { class: 'badge' }, '目前使用') : null,
        glyphImg(c.src, L)
      )
    );
  });
  const desc = images.length > 1
    ? `這次拍到 ${images.length} 個「${L}」${existing ? '，加上原本已建立的那一個' : ''}。點一下你最喜歡的那個保留：`
    : `「${L}」已經建立過了，要保留哪一個呢？`;
  const keepAll = offerKeepAll && existing && images.length === 1;
  const r = await showModal({
    title: `重複的字：${L}`,
    body: el('div', {}, el('p', {}, desc), grid),
    actions: keepAll ? [{ label: '其餘已建立的字都保留原本的', value: 'keep-all' }] : [],
    dismissible: false,
  });
  if (r === 'keep-all') return 'keep-all';
  const chosen = choices[(r && r.pick) || 0];
  if (chosen.isExisting) return false;
  current.glyphs[L] = chosen.src;
  return true;
}

/* =========================================================
 * 輸出字型 & 預覽
 * ======================================================= */
$('#btn-export').addEventListener('click', async () => {
  if (!current) return;
  const missing = LETTERS.filter((L) => !current.glyphs[L]);
  if (missing.length) {
    await showModal({
      title: '還有尚未建立的字體喔',
      body: el(
        'div',
        {},
        el('p', {}, `A–Z 還缺少 ${missing.length} 個字母，全部建立後才能輸出字型：`),
        el('div', { class: 'missing-list' }, missing.map((L) => el('span', {}, L)))
      ),
      actions: [{ label: '返回', value: true, class: 'btn-primary' }],
    });
    return; // 不進行輸出
  }
  const btn = $('#btn-export');
  btn.disabled = true;
  btn.textContent = '產生中…';
  try {
    if (typeof opentype === 'undefined') throw new Error('字型引擎載入失敗，請確認網路連線後重新整理');
    const font = await buildFont(current, { requireAll: true });
    const blob = fontToBlob(font);
    const filename = `${safeFileName(current.name)}.otf`;
    downloadBlob(blob, filename);
    await showModal({
      title: '字型已輸出！',
      body: el(
        'div',
        {},
        el('p', {}, `已下載「${filename}」。`),
        el('p', { class: 'small' }, '電腦：雙擊檔案即可安裝。iPhone / iPad：需透過支援安裝字型的 App（例如 iFont）安裝。Android：可在支援自訂字型的 App 中使用。大小寫都會顯示你的手寫字。')
      ),
      actions: [{ label: '好', value: true, class: 'btn-primary' }],
    });
  } catch (err) {
    console.error(err);
    toast('輸出失敗：' + (err.message || err), 4000);
  } finally {
    btn.disabled = false;
    btn.textContent = '輸出字型';
  }
});

function safeFileName(name) {
  return (name || 'FontAnytime').replace(/[\\/:*?"<>|]+/g, '_').trim() || 'FontAnytime';
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

let previewFontId = 0;
$('#btn-preview').addEventListener('click', async () => {
  if (!current) return;
  const count = LETTERS.filter((L) => current.glyphs[L]).length;
  if (!count) {
    toast('還沒有任何字母，先拍幾個字吧！');
    return;
  }
  if (typeof opentype === 'undefined') {
    toast('字型引擎載入失敗，請確認網路連線');
    return;
  }
  try {
    const font = await buildFont(current, { requireAll: false });
    const family = `fa-preview-${++previewFontId}`;
    const face = new FontFace(family, font.toArrayBuffer());
    await face.load();
    document.fonts.add(face);
    const box = el('div', { class: 'preview-box', style: `font-family:'${family}', var(--display)` });
    const input = el('input', {
      type: 'text',
      class: 'preview-input',
      value: 'HELLO WORLD',
      oninput: () => (box.textContent = input.value),
    });
    box.textContent = input.value;
    await showModal({
      title: '字型預覽',
      body: el(
        'div',
        {},
        el('p', { class: 'small' }, count < 26 ? `目前有 ${count} / 26 個字母，尚未建立的字會用系統字型顯示。` : '輸入文字看看效果：'),
        input,
        box
      ),
      actions: [{ label: '關閉', value: true, class: 'btn-primary' }],
    });
    document.fonts.delete(face);
  } catch (err) {
    console.error(err);
    toast('預覽失敗：' + (err.message || err));
  }
});

/* ========================================================= */
route();
