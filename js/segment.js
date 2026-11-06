// 從照片中把手寫字一個一個切割出來
// 流程：縮圖 → 灰階 → 自適應二值化 → 連通元件 → 合併同一個字的筆畫 → 依閱讀順序排序 → 裁切

const MAX_SIDE = 1600; // 處理用的最大邊長
export const GLYPH_SIZE = 256; // 儲存字形圖的最大邊長

/** 把影像來源（video / img / canvas）畫進 canvas 並縮到合適大小 */
export function toWorkingCanvas(source, srcW, srcH) {
  const scale = Math.min(1, MAX_SIDE / Math.max(srcW, srcH));
  const w = Math.round(srcW * scale);
  const h = Math.round(srcH * scale);
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, w, h);
  return c;
}

/** Bradley 自適應二值化：對光線不均的手機照片比較穩定 */
function binarize(imageData) {
  const { width: w, height: h, data } = imageData;
  const gray = new Float32Array(w * h);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  // 積分圖
  const W1 = w + 1;
  const integral = new Float64Array(W1 * (h + 1));
  for (let y = 0; y < h; y++) {
    let rowSum = 0;
    for (let x = 0; x < w; x++) {
      rowSum += gray[y * w + x];
      integral[(y + 1) * W1 + (x + 1)] = integral[y * W1 + (x + 1)] + rowSum;
    }
  }
  const half = Math.max(8, Math.round(Math.max(w, h) / 16));
  const T = 0.18; // 比周遭平均暗 18% 視為墨跡
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - half);
    const y1 = Math.min(h, y + half + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - half);
      const x1 = Math.min(w, x + half + 1);
      const count = (x1 - x0) * (y1 - y0);
      const sum =
        integral[y1 * W1 + x1] - integral[y0 * W1 + x1] - integral[y1 * W1 + x0] + integral[y0 * W1 + x0];
      const mean = sum / count;
      const g = gray[y * w + x];
      if (g < mean * (1 - T) && mean - g > 14) mask[y * w + x] = 1;
    }
  }
  return mask;
}

/** 8 連通元件標記 */
function labelComponents(mask, w, h) {
  const labels = new Int32Array(w * h);
  const comps = [];
  const stack = new Int32Array(w * h);
  let next = 1;
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || labels[start]) continue;
    const id = next++;
    let sp = 0;
    stack[sp++] = start;
    labels[start] = id;
    let minX = w, minY = h, maxX = 0, maxY = 0, area = 0;
    while (sp > 0) {
      const p = stack[--sp];
      const x = p % w;
      const y = (p - x) / w;
      area++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= w) continue;
          const q = ny * w + nx;
          if (mask[q] && !labels[q]) {
            labels[q] = id;
            stack[sp++] = q;
          }
        }
      }
    }
    comps.push({ ids: [id], minX, minY, maxX, maxY, area });
  }
  return { labels, comps };
}

const bw = (b) => b.maxX - b.minX + 1;
const bh = (b) => b.maxY - b.minY + 1;

function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function shouldMerge(a, b, refH) {
  const gap = refH * 0.06;
  const unionH = Math.max(a.maxY, b.maxY) - Math.min(a.minY, b.minY) + 1;
  const bothBig = bh(a) > refH * 0.55 && bh(b) > refH * 0.55;
  // 合併後比一個字高很多 → 是上下排得很近的兩個字，不要合併
  if (bothBig && unionH > refH * 1.4) return false;

  const overlapX = Math.min(a.maxX, b.maxX) - Math.max(a.minX, b.minX);
  const overlapY = Math.min(a.maxY, b.maxY) - Math.max(a.minY, b.minY);
  // 外框（加一點間距）有重疊
  if (overlapX > -gap && overlapY > -gap) {
    // 兩個都很大、左右並排：合起來仍只有一個字寬（例如橫線斷掉的 H）才合併，
    // 否則是相鄰的兩個字
    const minH = Math.min(bh(a), bh(b));
    if (bothBig && overlapY > minH * 0.5) {
      const unionW = Math.max(a.maxX, b.maxX) - Math.min(a.minX, b.minX) + 1;
      return unionW <= refH * 1.05;
    }
    return true;
  }
  // 上下堆疊的筆畫（例如斷掉的筆畫）：水平大致重疊、垂直很近，且合起來仍是一個字的高度
  const smallerW = Math.min(bw(a), bw(b));
  if (overlapX > smallerW * 0.6 && overlapY > -refH * 0.25 && unionH <= refH * 1.3) return true;
  return false;
}

function mergeBoxes(comps, refH) {
  let groups = comps.map((c) => ({ ...c, ids: [...c.ids] }));
  let changed = true;
  while (changed) {
    changed = false;
    outer: for (let i = 0; i < groups.length; i++) {
      for (let j = i + 1; j < groups.length; j++) {
        const a = groups[i];
        const b = groups[j];
        if (shouldMerge(a, b, refH)) {
          groups[i] = {
            ids: a.ids.concat(b.ids),
            minX: Math.min(a.minX, b.minX),
            minY: Math.min(a.minY, b.minY),
            maxX: Math.max(a.maxX, b.maxX),
            maxY: Math.max(a.maxY, b.maxY),
            area: a.area + b.area,
          };
          groups.splice(j, 1);
          changed = true;
          break outer;
        }
      }
    }
  }
  return groups;
}

/** 依閱讀順序（由上而下、由左而右）排序 */
function readingOrder(groups, refH) {
  const sorted = [...groups].sort((a, b) => a.minY + a.maxY - (b.minY + b.maxY));
  const rows = [];
  for (const g of sorted) {
    const cy = (g.minY + g.maxY) / 2;
    const row = rows.find((r) => Math.abs(r.cy - cy) < refH * 0.6);
    if (row) {
      row.items.push(g);
      row.cy = row.items.reduce((s, it) => s + (it.minY + it.maxY) / 2, 0) / row.items.length;
    } else {
      rows.push({ cy, items: [g] });
    }
  }
  rows.sort((a, b) => a.cy - b.cy);
  return rows.flatMap((r) => r.items.sort((a, b) => a.minX - b.minX));
}

/**
 * 切割照片中的每一個字
 * @param {HTMLCanvasElement} canvas 已縮放的工作 canvas
 * @returns {{box:{x,y,w,h}, glyph:HTMLCanvasElement, raw:HTMLCanvasElement}[]}
 */
export function segmentCharacters(canvas) {
  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const imageData = ctx.getImageData(0, 0, w, h);
  const mask = binarize(imageData);
  const { labels, comps } = labelComponents(mask, w, h);

  const minArea = Math.max(12, (w * h) / 40000);
  const edge = 2;
  let candidates = comps.filter((c) => {
    if (c.area < minArea) return false;
    // 碰到照片邊緣的通常是紙張邊緣或陰影
    if (c.minX < edge || c.minY < edge || c.maxX >= w - edge || c.maxY >= h - edge) return false;
    // 太大的區塊不是字
    if (bw(c) > w * 0.6 || bh(c) > h * 0.6) return false;
    return true;
  });
  if (!candidates.length) return [];

  // 以「較大元件」的高度作為字高參考
  const maxH = Math.max(...candidates.map(bh));
  let refH = median(candidates.filter((c) => bh(c) > maxH * 0.3).map(bh)) || maxH;

  let groups = mergeBoxes(candidates, refH);
  refH = median(groups.filter((g) => bh(g) > maxH * 0.3).map(bh)) || refH;

  // 去掉雜點：太小或極細的區塊
  groups = groups.filter((g) => {
    const gw = bw(g);
    const gh = bh(g);
    if (gh < refH * 0.3 && gw < refH * 0.3) return false;
    if (g.area < (refH * refH) / 200) return false;
    return true;
  });

  groups = readingOrder(groups, refH);

  return groups.map((g) => {
    const idSet = new Set(g.ids);
    const gw = bw(g);
    const gh = bh(g);
    // 只保留屬於這個字的像素，避免把旁邊字的筆畫帶進來
    const raw = document.createElement('canvas');
    raw.width = gw;
    raw.height = gh;
    const rctx = raw.getContext('2d');
    const out = rctx.createImageData(gw, gh);
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const l = labels[(y + g.minY) * w + (x + g.minX)];
        if (l && idSet.has(l)) {
          const o = (y * gw + x) * 4;
          out.data[o] = 17;
          out.data[o + 1] = 17;
          out.data[o + 2] = 17;
          out.data[o + 3] = 255;
        }
      }
    }
    rctx.putImageData(out, 0, 0);

    return {
      box: { x: g.minX, y: g.minY, w: gw, h: gh },
      glyph: makeGlyphCanvas(raw),
      raw,
    };
  });
}

/** 放大/縮小到固定大小並平滑邊緣（模糊後再二值化），作為儲存與向量化的字形 */
function makeGlyphCanvas(raw) {
  const scale = GLYPH_SIZE / Math.max(raw.width, raw.height);
  const tw = Math.max(1, Math.round(raw.width * scale));
  const th = Math.max(1, Math.round(raw.height * scale));
  const pad = 4;
  const c = document.createElement('canvas');
  c.width = tw + pad * 2;
  c.height = th + pad * 2;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const blur = Math.max(0.6, scale * 0.45);
  ctx.filter = `blur(${blur.toFixed(2)}px)`;
  ctx.drawImage(raw, pad, pad, tw, th);
  ctx.filter = 'none';
  const img = ctx.getImageData(0, 0, c.width, c.height);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const on = d[i + 3] >= 128;
    d[i] = d[i + 1] = d[i + 2] = 17;
    d[i + 3] = on ? 255 : 0;
  }
  ctx.putImageData(img, 0, 0);
  return trimCanvas(c);
}

/** 去掉透明邊 */
export function trimCanvas(c) {
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const { data, width: w, height: h } = ctx.getImageData(0, 0, c.width, c.height);
  let minX = w, minY = h, maxX = -1, maxY = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (data[(y * w + x) * 4 + 3] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return c;
  const out = document.createElement('canvas');
  out.width = maxX - minX + 1;
  out.height = maxY - minY + 1;
  out.getContext('2d').drawImage(c, -minX, -minY);
  return out;
}
