// 手寫字母辨識：在 EMNIST（約 14 萬個真人手寫字母）上訓練的卷積神經網路（CNN）
// 模型權重放在 model/letters.bin（int8 量化，約 300 KB），完全在瀏覽器端執行，不需要網路服務。
//
// 前處理盡量模仿 EMNIST 的格式：
//   1. 把字縮放到固定大小、量測筆畫粗細，並用膨脹 / 侵蝕把筆畫調成 EMNIST 的粗細
//   2. 依外框置中，最長邊佔 28×28 中的 22 px
//   3. 白字黑底、0~1 灰階

const MODEL_URL = new URL('../model/', import.meta.url);
const WORK = 96; // 調整筆畫用的工作尺寸
const TARGET_STROKE = 0.105; // EMNIST 筆畫寬度 ÷ 字的最長邊（中位數）
const FIT = 22; // 最長邊在 28×28 中佔的像素
const ACCEPT_PROB = 0.8; // 低於這個信心值就放到「未辨識」區，讓使用者從前 3 名一鍵確認

let modelPromise = null;

export function loadModel() {
  if (modelPromise) return modelPromise;
  modelPromise = (async () => {
    const [meta, buf] = await Promise.all([
      fetch(new URL('letters.json', MODEL_URL)).then((r) => {
        if (!r.ok) throw new Error('模型載入失敗');
        return r.json();
      }),
      fetch(new URL('letters.bin', MODEL_URL)).then((r) => {
        if (!r.ok) throw new Error('模型載入失敗');
        return r.arrayBuffer();
      }),
    ]);
    const layers = meta.layers.map((L) => {
      const q = new Int8Array(buf, L.w[0], L.w[1]);
      const scale = new Float32Array(buf, L.scale[0], L.scale[1] / 4);
      const b = new Float32Array(buf, L.b[0], L.b[1] / 4);
      const cout = L.shape[L.shape.length - 1];
      const w = new Float32Array(q.length);
      for (let i = 0; i < q.length; i++) w[i] = q[i] * scale[i % cout];
      return { ...L, w, b: Float32Array.from(b), cout, cin: L.type === 'conv' ? L.shape[2] : L.shape[0] };
    });
    return { labels: meta.labels, layers };
  })();
  modelPromise.catch(() => (modelPromise = null));
  return modelPromise;
}

/* ---------------- CNN 推論（NHWC） ---------------- */

function conv3x3(input, H, W, L) {
  const { cin, cout, w, b } = L;
  const out = new Float32Array(H * W * cout);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * cout;
      out.set(b, o);
      for (let ky = 0; ky < 3; ky++) {
        const iy = y + ky - 1;
        if (iy < 0 || iy >= H) continue;
        for (let kx = 0; kx < 3; kx++) {
          const ix = x + kx - 1;
          if (ix < 0 || ix >= W) continue;
          const ib = (iy * W + ix) * cin;
          const wb = (ky * 3 + kx) * cin * cout;
          for (let ci = 0; ci < cin; ci++) {
            const v = input[ib + ci];
            if (v === 0) continue; // ReLU 之後很多 0，跳過可以快不少
            const wo = wb + ci * cout;
            for (let co = 0; co < cout; co++) out[o + co] += v * w[wo + co];
          }
        }
      }
    }
  }
  return out;
}

function relu(a) {
  for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0;
  return a;
}

function maxPool2(input, H, W, C) {
  const h = H >> 1, w = W >> 1;
  const out = new Float32Array(h * w * C);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      for (let c = 0; c < C; c++) {
        const i0 = ((2 * y) * W + 2 * x) * C + c;
        const i1 = i0 + W * C;
        out[(y * w + x) * C + c] = Math.max(input[i0], input[i0 + C], input[i1], input[i1 + C]);
      }
    }
  }
  return out;
}

function dense(input, L) {
  const { cin, cout, w, b } = L;
  const out = Float32Array.from(b);
  for (let i = 0; i < cin; i++) {
    const v = input[i];
    if (v === 0) continue;
    const wo = i * cout;
    for (let o = 0; o < cout; o++) out[o] += v * w[wo + o];
  }
  return out;
}

function runModel(model, x) {
  let h = x, H = 28, W = 28;
  for (const L of model.layers) {
    if (L.type === 'conv') {
      h = conv3x3(h, H, W, L);
      if (L.relu) relu(h);
      if (L.pool) {
        h = maxPool2(h, H, W, L.cout);
        H >>= 1;
        W >>= 1;
      }
    } else {
      h = dense(h, L);
      if (L.relu) relu(h);
    }
  }
  // softmax
  const m = Math.max(...h);
  let sum = 0;
  const p = h.map((v) => {
    const e = Math.exp(v - m);
    sum += e;
    return e;
  });
  return p.map((v) => v / sum);
}

/* ---------------- 前處理 ---------------- */

function alphaMask(canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const m = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < m.length; i++) m[i] = data[i * 4 + 3] >= 128 ? 1 : 0;
  return m;
}

function strokeWidth(m, w, h) {
  let area = 0, per = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = m[y * w + x];
      area += v;
      const r = x + 1 < w ? m[y * w + x + 1] : 0;
      const d = y + 1 < h ? m[(y + 1) * w + x] : 0;
      if (v !== r) per++;
      if (v !== d) per++;
      if (x === 0 && v) per++;
      if (y === 0 && v) per++;
    }
  }
  return per ? (2 * area) / per : 0;
}

/** 以圓形結構元素做膨脹（grow=true）或侵蝕 */
function morph(m, w, h, r, grow) {
  const out = new Uint8Array(w * h);
  const offs = [];
  for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= r * r + r * 0.5) offs.push([dx, dy]);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let hit = !grow;
      for (const [dx, dy] of offs) {
        const nx = x + dx, ny = y + dy;
        const v = nx >= 0 && ny >= 0 && nx < w && ny < h ? m[ny * w + nx] : 0;
        if (grow ? v : !v) { hit = grow; break; }
      }
      out[y * w + x] = hit ? 1 : 0;
    }
  }
  return out;
}

/**
 * 把切割出來的字（黑字透明底 canvas）轉成模型輸入
 * @returns {Float32Array} 28×28 灰階，白字 = 1
 */
export function preprocess(raw) {
  // 1. 縮放到工作尺寸（最長邊 = WORK），四周留空間給膨脹
  const s = WORK / Math.max(raw.width, raw.height);
  const pad = 16;
  const ww = Math.max(1, Math.round(raw.width * s)) + pad * 2;
  const wh = Math.max(1, Math.round(raw.height * s)) + pad * 2;
  const c = document.createElement('canvas');
  c.width = ww;
  c.height = wh;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(raw, pad, pad, ww - pad * 2, wh - pad * 2);
  let m = alphaMask(c);

  // 2. 筆畫粗細正規化
  const sw = strokeWidth(m, ww, wh);
  const target = TARGET_STROKE * WORK;
  if (sw > 0 && sw < target * 0.8) {
    m = morph(m, ww, wh, Math.min(8, Math.round((target - sw) / 2)), true);
  } else if (sw > target * 1.5) {
    const r = Math.min(6, Math.round((sw - target) / 2));
    const eroded = morph(m, ww, wh, r, false);
    const before = m.reduce((a, v) => a + v, 0);
    const after = eroded.reduce((a, v) => a + v, 0);
    if (after > before * 0.3) m = eroded;
  }

  // 3. 取外框
  let minX = ww, minY = wh, maxX = -1, maxY = -1;
  for (let y = 0; y < wh; y++) for (let x = 0; x < ww; x++) if (m[y * ww + x]) {
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const out = new Float32Array(28 * 28);
  if (maxX < 0) return out;
  const bw = maxX - minX + 1, bh = maxY - minY + 1;

  // 4. 畫成白字的 canvas，縮小到 28×28 並置中
  const img = ctx.createImageData(ww, wh);
  for (let i = 0; i < m.length; i++) {
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = 255;
    img.data[i * 4 + 3] = m[i] ? 255 : 0;
  }
  ctx.putImageData(img, 0, 0);
  const k = FIT / Math.max(bw, bh);
  const tw = bw * k, th = bh * k;
  // 分兩段縮小，邊緣比較平滑（接近 EMNIST 的灰階邊）
  const mid = document.createElement('canvas');
  mid.width = Math.max(1, Math.round(tw * 2));
  mid.height = Math.max(1, Math.round(th * 2));
  const mctx = mid.getContext('2d');
  mctx.imageSmoothingEnabled = true;
  mctx.imageSmoothingQuality = 'high';
  mctx.drawImage(c, minX, minY, bw, bh, 0, 0, mid.width, mid.height);
  const fin = document.createElement('canvas');
  fin.width = fin.height = 28;
  const fctx = fin.getContext('2d', { willReadFrequently: true });
  fctx.imageSmoothingEnabled = true;
  fctx.imageSmoothingQuality = 'high';
  fctx.drawImage(mid, (28 - tw) / 2, (28 - th) / 2, tw, th);
  const d = fctx.getImageData(0, 0, 28, 28).data;
  for (let i = 0; i < 784; i++) out[i] = d[i * 4 + 3] / 255;
  return out;
}

/** 筆跡是不是一條（可能有點斜的）細長直線：用墨跡像素的主軸長寬比判斷 */
function isBar(raw) {
  const w = raw.width, h = raw.height;
  const m = alphaMask(raw);
  let n = 0, sx = 0, sy = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) { n++; sx += x; sy += y; }
  if (!n) return false;
  const mx = sx / n, my = sy / n;
  let cxx = 0, cyy = 0, cxy = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (m[y * w + x]) {
    const dx = x - mx, dy = y - my;
    cxx += dx * dx; cyy += dy * dy; cxy += dx * dy;
  }
  const tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(1e-6, tr / 2 - disc);
  return Math.sqrt(l1 / l2) > 5.5; // 長軸 / 短軸
}

/**
 * 辨識一個字
 * @param {HTMLCanvasElement} raw 黑字透明底
 * @returns {Promise<{letter: string|null, prob: number, top: {letter:string, prob:number}[]}>}
 */
export async function recognize(raw) {
  const model = await loadModel();
  const probs = runModel(model, preprocess(raw));
  const top = [...probs]
    .map((p, i) => ({ letter: model.labels[i], prob: p }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 3);
  // 大寫字母中只有 I 是「一條線」（EMNIST 的 L 類別包含小寫 l，模型容易把直線判成 L）
  if (isBar(raw) && ['L', 'I', 'J', 'T'].includes(top[0].letter)) {
    const i = top.findIndex((t) => t.letter === 'I');
    const iEntry = i >= 0 ? top.splice(i, 1)[0] : { letter: 'I', prob: top[0].prob };
    top.unshift({ letter: 'I', prob: Math.max(iEntry.prob, top[0].prob) });
    top.length = Math.min(top.length, 3);
  }
  const best = top[0];
  return { letter: best.prob >= ACCEPT_PROB ? best.letter : null, prob: best.prob, top };
}

/**
 * 依序辨識多個字
 * @param {HTMLCanvasElement[]} raws
 * @param {(done:number, total:number, index:number, letter:string|null)=>void} onProgress
 */
export async function recognizeAll(raws, onProgress) {
  await loadModel();
  const results = [];
  for (let i = 0; i < raws.length; i++) {
    const r = await recognize(raws[i]);
    results.push(r);
    onProgress && onProgress(i + 1, raws.length, i, r.letter);
    // 讓畫面有機會更新
    if (i % 4 === 3) await new Promise((res) => setTimeout(res, 0));
  }
  return results;
}
