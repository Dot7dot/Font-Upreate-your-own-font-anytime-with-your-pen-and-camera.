// 以 IndexedDB 儲存資料夾（每個資料夾 = 一套字體）
// Folder 結構：
// { id, name, createdAt, updatedAt,
//   glyphs: { A: dataURL, B: dataURL, ... } }  // 已確認的字形（黑字透明底 PNG）

const DB_NAME = 'font-anytime';
const STORE = 'folders';
let dbPromise = null;

function openDB() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx(mode, fn) {
  return openDB().then(
    (db) =>
      new Promise((resolve, reject) => {
        const t = db.transaction(STORE, mode);
        const store = t.objectStore(STORE);
        const result = fn(store);
        t.oncomplete = () => resolve(result && 'result' in result ? result.result : result);
        t.onerror = () => reject(t.error);
        t.onabort = () => reject(t.error);
      })
  );
}

export function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export async function listFolders() {
  const all = await tx('readonly', (s) => s.getAll());
  return (all || []).sort((a, b) => b.updatedAt - a.updatedAt);
}

export function getFolder(id) {
  return tx('readonly', (s) => s.get(id));
}

export function saveFolder(folder) {
  folder.updatedAt = Date.now();
  return tx('readwrite', (s) => s.put(folder));
}

export function deleteFolder(id) {
  return tx('readwrite', (s) => s.delete(id));
}

export async function createFolder(name) {
  const folder = {
    id: uid(),
    name,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    glyphs: {},
  };
  await saveFolder(folder);
  return folder;
}
