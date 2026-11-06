import gzip, numpy as np, os
D = os.environ.get('EMNIST_DIR', os.path.join(os.path.dirname(__file__), 'emnist'))
def load(split):
    x = np.frombuffer(gzip.open(f'{D}/emnist-letters-{split}-images-idx3-ubyte.gz').read()[16:], np.uint8).reshape(-1, 28, 28)
    y = np.frombuffer(gzip.open(f'{D}/emnist-letters-{split}-labels-idx1-ubyte.gz').read()[8:], np.uint8).astype(np.int32) - 1
    return np.ascontiguousarray(x.transpose(0, 2, 1)), y   # EMNIST 影像是轉置存放的
