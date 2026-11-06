// 單字 OCR：使用 Tesseract.js（從 CDN 載入），只辨識大寫 A-Z
// 手寫字很難一次辨識成功，所以採「多輪」策略：
//   第 1 輪：單字模式（PSM 10）
//   第 2 輪：沒辨識出來的，改用區塊模式（PSM 6）並放大
//   第 3 輪：單字詞模式（PSM 8）
//   最後：又細又長的直線視為 I
/* global Tesseract */

const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const PASSES = [
  { psm: '10', height: 72, pad: 28, minConf: 10 },
  { psm: '6', height: 120, pad: 40, minConf: 10 },
  { psm: '8', height: 72, pad: 28, minConf: 0 },
];
let workerPromise = null;

export function getWorker(onStatus) {
  if (workerPromise) return workerPromise;
  if (typeof Tesseract === 'undefined') {
    return Promise.reject(new Error('OCR 引擎載入失敗，請確認網路連線後重新整理頁面'));
  }
  workerPromise = (async () => {
    onStatus && onStatus('正在載入文字辨識引擎…（第一次需要幾秒鐘）');
    const worker = await Tesseract.createWorker('eng', 1);
    await worker.setParameters({ tessedit_char_whitelist: LETTERS });
    return worker;
  })();
  workerPromise.catch(() => {
    workerPromise = null;
  });
  return workerPromise;
}

/** OCR 用：白底黑字、四周留白、固定字高 */
function ocrCanvas(raw, height, pad) {
  const scale = height / Math.max(raw.height, raw.width * 0.8);
  const tw = Math.max(1, Math.round(raw.width * scale));
  const th = Math.max(1, Math.round(raw.height * scale));
  const c = document.createElement('canvas');
  c.width = tw + pad * 2;
  c.height = th + pad * 2;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(raw, pad, pad, tw, th);
  return c;
}

/**
 * 辨識多個切割出來的字
 * @param {HTMLCanvasElement[]} raws 黑字透明底的字形
 * @param {(done:number,total:number,index:number,letter:string|null)=>void} onProgress
 * @returns {Promise<(string|null)[]>}
 */
export async function recognizeAll(raws, onProgress) {
  const worker = await getWorker();
  const result = raws.map(() => null);
  let done = 0;
  for (let p = 0; p < PASSES.length; p++) {
    const pass = PASSES[p];
    const todo = result.map((r, i) => (r ? -1 : i)).filter((i) => i >= 0);
    if (!todo.length) break;
    await worker.setParameters({ tessedit_pageseg_mode: pass.psm });
    for (const i of todo) {
      const { data } = await worker.recognize(ocrCanvas(raws[i], pass.height, pass.pad));
      const text = (data.text || '').toUpperCase().replace(/[^A-Z]/g, '');
      if (text.length === 1 && (data.confidence || 0) >= pass.minConf) result[i] = text;
      if (p === 0 || result[i]) {
        if (p === 0) done++;
        onProgress && onProgress(done, raws.length, i, result[i]);
      }
    }
  }
  // 又細又長的一直線：大寫字母裡只有 I
  result.forEach((r, i) => {
    if (!r && raws[i].width / raws[i].height < 0.22) {
      result[i] = 'I';
      onProgress && onProgress(done, raws.length, i, 'I');
    }
  });
  return result;
}
