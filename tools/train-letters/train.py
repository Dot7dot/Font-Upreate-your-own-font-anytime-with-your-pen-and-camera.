"""Train a small CNN on EMNIST letters (26 classes) and export int8 weights for the browser."""
import json, sys, time, os
import numpy as np
import jax, jax.numpy as jnp, optax
from scipy import ndimage
from multiprocessing import Pool
from data import load

EPOCHS = int(os.environ.get('EPOCHS', 14))
BATCH = 256
OUT = sys.argv[1] if len(sys.argv) > 1 else 'out'
os.makedirs(OUT, exist_ok=True)

# ---------------- model ----------------
# conv(1->32) conv(32->32) pool | conv(32->64) conv(64->64) pool | conv(64->128) pool | dense(1152->128) dense(128->26)
CONVS = [(1, 32), (32, 32), (32, 64), (64, 64), (64, 128)]
POOL_AFTER = {1, 3, 4}
DENSE = [(3 * 3 * 128, 128), (128, 26)]

def init(key):
    params = []
    for cin, cout in CONVS:
        key, k = jax.random.split(key)
        params.append({'w': jax.random.normal(k, (3, 3, cin, cout)) * np.sqrt(2 / (9 * cin)), 'b': jnp.zeros(cout)})
    for nin, nout in DENSE:
        key, k = jax.random.split(key)
        params.append({'w': jax.random.normal(k, (nin, nout)) * np.sqrt(2 / nin), 'b': jnp.zeros(nout)})
    return params

def forward(params, x, key=None):
    h = x
    for i in range(len(CONVS)):
        p = params[i]
        h = jax.lax.conv_general_dilated(h, p['w'], (1, 1), 'SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC')) + p['b']
        h = jax.nn.relu(h)
        if i in POOL_AFTER:
            h = jax.lax.reduce_window(h, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')
    h = h.reshape(h.shape[0], -1)
    p = params[len(CONVS)]
    h = jax.nn.relu(h @ p['w'] + p['b'])
    if key is not None:
        keep = jax.random.bernoulli(key, 0.6, h.shape)
        h = jnp.where(keep, h / 0.6, 0)
    p = params[len(CONVS) + 1]
    return h @ p['w'] + p['b']

def loss_fn(params, x, y, key):
    logits = forward(params, x, key)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

# ---------------- augmentation ----------------
def augment_one(args):
    img, seed = args
    rng = np.random.default_rng(seed)
    im = img.astype(np.float32) / 255.0
    # 筆畫粗細：photo 拍到的筆跡可能比 EMNIST 細或粗
    r = rng.random()
    if r < 0.25:
        im = ndimage.grey_erosion(im, size=(2, 2))
    elif r < 0.5:
        im = ndimage.grey_dilation(im, size=(2, 2))
    elif r < 0.58:
        im = ndimage.grey_dilation(im, size=(3, 3))
    # 仿射：旋轉、縮放、傾斜、平移
    ang = np.deg2rad(rng.uniform(-14, 14))
    sc = rng.uniform(0.82, 1.12)
    sx, sy = sc * rng.uniform(0.9, 1.1), sc * rng.uniform(0.9, 1.1)
    sh = rng.uniform(-0.25, 0.25)
    c, s = np.cos(ang), np.sin(ang)
    A = np.array([[c, -s], [s, c]]) @ np.array([[1, sh], [0, 1]]) @ np.diag([sy, sx])
    Ainv = np.linalg.inv(A)
    center = np.array([13.5, 13.5])
    t = rng.uniform(-2, 2, 2)
    offset = center - Ainv @ (center + t)
    im = ndimage.affine_transform(im, Ainv, offset=offset, order=1, mode='constant')
    # 偶爾變得比較「二值」或比較模糊
    r = rng.random()
    if r < 0.2:
        im = (im > 0.45).astype(np.float32)
        im = ndimage.gaussian_filter(im, 0.5)
    elif r < 0.3:
        im = ndimage.gaussian_filter(im, 0.8)
    return np.clip(im, 0, 1)

def augment(x, pool, epoch):
    seeds = np.arange(len(x)) + epoch * 1_000_003
    return np.stack(pool.map(augment_one, zip(x, seeds), chunksize=2048))

# ---------------- training ----------------
def main():
    xtr, ytr = load('train')
    xte, yte = load('test')
    xte_f = (xte.astype(np.float32) / 255.0)[..., None]
    params = init(jax.random.PRNGKey(0))
    steps = EPOCHS * (len(xtr) // BATCH)
    sched = optax.warmup_cosine_decay_schedule(0, 2e-3, 300, steps, 1e-5)
    opt = optax.adamw(sched, weight_decay=1e-4)
    state = opt.init(params)

    @jax.jit
    def step(params, state, x, y, key):
        l, g = jax.value_and_grad(loss_fn)(params, x, y, key)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, l

    @jax.jit
    def predict(params, x):
        return forward(params, x).argmax(-1)

    def evaluate(params):
        preds = np.concatenate([np.asarray(predict(params, xte_f[i:i + 2000])) for i in range(0, len(xte_f), 2000)])
        return (preds == yte).mean()

    key = jax.random.PRNGKey(1)
    with Pool(3) as pool:
        for ep in range(EPOCHS):
            t0 = time.time()
            xa = augment(xtr, pool, ep)[..., None]
            perm = np.random.default_rng(ep).permutation(len(xa))
            losses = []
            for i in range(0, len(perm) - BATCH + 1, BATCH):
                idx = perm[i:i + BATCH]
                key, k = jax.random.split(key)
                params, state, l = step(params, state, xa[idx], ytr[idx], k)
                losses.append(float(l))
            acc = evaluate(params)
            print(f'epoch {ep + 1}/{EPOCHS} loss {np.mean(losses):.4f} test acc {acc:.4f} ({time.time() - t0:.0f}s)', flush=True)

    np.save(f'{OUT}/params.npy', np.array(jax.tree.map(np.asarray, params), dtype=object), allow_pickle=True)
    export(params, f'{OUT}', evaluate)

def export(params, out, evaluate):
    """int8 per-output-channel quantization → letters.bin + letters.json"""
    layers, blobs, offset = [], [], 0
    qparams = []
    for i, p in enumerate(params):
        w = np.asarray(p['w'], np.float32)
        b = np.asarray(p['b'], np.float32)
        flat = w.reshape(-1, w.shape[-1])
        scale = np.abs(flat).max(0) / 127.0 + 1e-12
        q = np.round(flat / scale).astype(np.int8)
        qparams.append({'w': jnp.asarray((q * scale).reshape(w.shape)), 'b': jnp.asarray(b)})
        kind = 'conv' if i < len(CONVS) else 'dense'
        entry = {'type': kind, 'shape': list(w.shape), 'pool': i in POOL_AFTER, 'relu': i != len(params) - 1}
        for name, arr in (('w', q.tobytes()), ('scale', scale.astype(np.float32).tobytes()), ('b', b.tobytes())):
            pad = (-offset) % 4
            blobs.append(b'\0' * pad); offset += pad
            entry[name] = [offset, len(arr)]
            blobs.append(arr); offset += len(arr)
        layers.append(entry)
    with open(f'{out}/letters.bin', 'wb') as f:
        f.write(b''.join(blobs))
    json.dump({'input': [28, 28, 1], 'labels': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'layers': layers}, open(f'{out}/letters.json', 'w'))
    print('float acc', evaluate(params), 'int8 acc', evaluate(qparams), 'bytes', offset)

if __name__ == '__main__':
    main()
