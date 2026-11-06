// 把字形點陣圖描邊成向量輪廓，並用 opentype.js 組成真正的 OpenType 字型檔
/* global opentype */

export const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

const UNITS_PER_EM = 1000;
const CAP_HEIGHT = 700;
const ASCENDER = 880;
const DESCENDER = -220;
const SIDE_BEARING = 60;

/* ---------- 點陣圖 → 輪廓 ---------- */

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

/** 取得字形遮罩（1 = 墨跡） */
async function maskFromDataURL(dataURL) {
  const img = await loadImage(dataURL);
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, c.width, c.height);
  const mask = new Uint8Array(c.width * c.height);
  for (let i = 0; i < mask.length; i++) mask[i] = data[i * 4 + 3] >= 128 ? 1 : 0;
  return { mask, w: c.width, h: c.height };
}

/**
 * 沿著像素邊界描出所有封閉輪廓。
 * 外框為螢幕座標的順時針方向、內洞為逆時針方向（翻轉 y 軸後剛好符合 PostScript 字型的規範）。
 * 回傳的點是每段單位邊的中點，可以消除鋸齒。
 */
export function traceContours(mask, w, h) {
  const fg = (x, y) => x >= 0 && y >= 0 && x < w && y < h && mask[y * w + x] === 1;
  const W1 = w + 1;
  const out = new Map(); // vertex -> [{to, dx, dy}]
  const add = (x0, y0, x1, y1) => {
    const k = y0 * W1 + x0;
    let list = out.get(k);
    if (!list) out.set(k, (list = []));
    list.push({ x0, y0, x1, y1, dx: x1 - x0, dy: y1 - y0, used: false });
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!fg(x, y)) continue;
      if (!fg(x, y - 1)) add(x, y, x + 1, y); // 上
      if (!fg(x + 1, y)) add(x + 1, y, x + 1, y + 1); // 右
      if (!fg(x, y + 1)) add(x + 1, y + 1, x, y + 1); // 下
      if (!fg(x - 1, y)) add(x, y + 1, x, y); // 左
    }
  }

  const contours = [];
  for (const list of out.values()) {
    for (const startEdge of list) {
      if (startEdge.used) continue;
      const pts = [];
      let e = startEdge;
      while (e && !e.used) {
        e.used = true;
        pts.push([(e.x0 + e.x1) / 2, (e.y0 + e.y1) / 2]);
        const nexts = (out.get(e.y1 * W1 + e.x1) || []).filter((n) => !n.used);
        if (nexts.length <= 1) {
          e = nexts[0];
        } else {
          // 鞍點：優先左轉（讓對角相連的像素連在一起，與 8 連通一致）
          const lx = e.dy, ly = -e.dx;
          e = nexts.find((n) => n.dx === lx && n.dy === ly) ||
              nexts.find((n) => n.dx === e.dx && n.dy === e.dy) ||
              nexts[0];
        }
      }
      if (pts.length >= 4) contours.push(pts);
    }
  }
  return contours;
}

/** Ramer–Douglas–Peucker 簡化（封閉多邊形） */
function simplifyClosed(pts, eps) {
  if (pts.length < 8) return pts;
  // 以離第一點最遠的點把多邊形分成兩段
  let far = 0, farD = -1;
  for (let i = 1; i < pts.length; i++) {
    const d = (pts[i][0] - pts[0][0]) ** 2 + (pts[i][1] - pts[0][1]) ** 2;
    if (d > farD) { farD = d; far = i; }
  }
  const a = rdp(pts.slice(0, far + 1), eps);
  const b = rdp(pts.slice(far).concat([pts[0]]), eps);
  return a.slice(0, -1).concat(b.slice(0, -1));
}

function rdp(pts, eps) {
  if (pts.length < 3) return pts;
  const [x1, y1] = pts[0];
  const [x2, y2] = pts[pts.length - 1];
  const dx = x2 - x1, dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1e-9;
  let idx = 0, maxD = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const d = Math.abs(dy * pts[i][0] - dx * pts[i][1] + x2 * y1 - y2 * x1) / len;
    if (d > maxD) { maxD = d; idx = i; }
  }
  if (maxD > eps) {
    const l = rdp(pts.slice(0, idx + 1), eps);
    const r = rdp(pts.slice(idx), eps);
    return l.slice(0, -1).concat(r);
  }
  return [pts[0], pts[pts.length - 1]];
}

function polygonArea(pts) {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const [x1, y1] = pts[i];
    const [x2, y2] = pts[(i + 1) % pts.length];
    a += x1 * y2 - x2 * y1;
  }
  return a / 2;
}

/** 由一張字形圖建立 opentype.Path 與字寬 */
export async function glyphPathFromImage(dataURL) {
  const { mask, w, h } = await maskFromDataURL(dataURL);
  const scale = CAP_HEIGHT / h;
  const eps = Math.max(0.6, h / 220);
  const minArea = (h * h) / 4000;
  const contours = traceContours(mask, w, h)
    .map((c) => simplifyClosed(c, eps))
    .filter((c) => c.length >= 3 && Math.abs(polygonArea(c)) >= minArea);

  const path = new opentype.Path();
  const tx = (x) => Math.round(SIDE_BEARING + x * scale);
  const ty = (y) => Math.round((h - y) * scale);

  for (const pts of contours) {
    const n = pts.length;
    const mid = (i) => {
      const p = pts[i % n];
      const q = pts[(i + 1) % n];
      return [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
    };
    // 以「邊的中點」為曲線端點、頂點為控制點，畫出平滑的二次曲線；
    // 轉角很尖的地方保持直線，保留筆畫的稜角
    const start = mid(n - 1);
    path.moveTo(tx(start[0]), ty(start[1]));
    for (let i = 0; i < n; i++) {
      const prev = pts[(i - 1 + n) % n];
      const cur = pts[i];
      const nxt = pts[(i + 1) % n];
      const v1x = cur[0] - prev[0], v1y = cur[1] - prev[1];
      const v2x = nxt[0] - cur[0], v2y = nxt[1] - cur[1];
      const cos = (v1x * v2x + v1y * v2y) / ((Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y)) || 1);
      const m = mid(i);
      if (cos < -0.2) {
        path.lineTo(tx(cur[0]), ty(cur[1]));
        path.lineTo(tx(m[0]), ty(m[1]));
      } else {
        path.quadraticCurveTo(tx(cur[0]), ty(cur[1]), tx(m[0]), ty(m[1]));
      }
    }
    path.closePath();
  }
  const advanceWidth = Math.round(w * scale + SIDE_BEARING * 2);
  return { path, advanceWidth };
}

/* ---------- 組字型 ---------- */

function asciiName(name, fallback) {
  const s = (name || '').normalize('NFKD').replace(/[^\x20-\x7E]/g, '').trim();
  return s || fallback;
}

/**
 * 依資料夾內容建立字型
 * @param {{name:string, glyphs:Object<string,string>}} folder
 * @param {{requireAll?: boolean}} opts
 */
export async function buildFont(folder, { requireAll = true } = {}) {
  const glyphs = [];
  glyphs.push(
    new opentype.Glyph({ name: '.notdef', unicode: undefined, advanceWidth: 500, path: notdefPath() })
  );
  glyphs.push(new opentype.Glyph({ name: 'space', unicode: 32, advanceWidth: 320, path: new opentype.Path() }));

  for (const L of LETTERS) {
    const src = folder.glyphs[L];
    if (!src) {
      if (requireAll) throw new Error(`缺少字母 ${L}`);
      continue;
    }
    const { path, advanceWidth } = await glyphPathFromImage(src);
    const g = new opentype.Glyph({ name: L, unicode: L.charCodeAt(0), advanceWidth, path });
    // 小寫也對應到同一個字形，打小寫字也看得到
    g.addUnicode(L.toLowerCase().charCodeAt(0));
    glyphs.push(g);
  }

  // 字型名稱可以用中文（Windows 名稱表是 UTF-16）；PostScript 名稱則必須是 ASCII
  const family = (folder.name || '').trim() || 'Font Anytime';
  const font = new opentype.Font({
    familyName: family,
    styleName: 'Regular',
    unitsPerEm: UNITS_PER_EM,
    ascender: ASCENDER,
    descender: DESCENDER,
    designer: 'Font Anytime',
    manufacturer: 'Font Anytime',
    version: '1.0',
    glyphs,
  });
  const psBase = asciiName(folder.name, '').replace(/[^A-Za-z0-9-]/g, '') || `FontAnytime${(folder.id || '').replace(/[^A-Za-z0-9]/g, '').slice(0, 8)}`;
  font.names.postScriptName = { en: `${psBase.slice(0, 50)}-Regular` };
  return font;
}

function notdefPath() {
  const p = new opentype.Path();
  p.moveTo(60, 0);
  p.lineTo(60, 700);
  p.lineTo(440, 700);
  p.lineTo(440, 0);
  p.closePath();
  p.moveTo(110, 50);
  p.lineTo(390, 50);
  p.lineTo(390, 650);
  p.lineTo(110, 650);
  p.closePath();
  return p;
}

export function fontToBlob(font) {
  return new Blob([font.toArrayBuffer()], { type: 'font/otf' });
}
