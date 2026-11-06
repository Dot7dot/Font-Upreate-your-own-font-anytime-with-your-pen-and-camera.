# 手寫字母辨識模型訓練

`model/letters.bin` / `model/letters.json` 是用這裡的腳本在 [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset)（約 14.5 萬個真人手寫字母、26 類）上訓練出來的小型 CNN，權重以 int8 量化。

## 重新訓練

```bash
pip install "jax[cpu]" optax scipy numpy
# 把下列 4 個檔案放到 tools/train-letters/emnist/（或用 EMNIST_DIR 指定資料夾）
#   emnist-letters-train-images-idx3-ubyte.gz  emnist-letters-train-labels-idx1-ubyte.gz
#   emnist-letters-test-images-idx3-ubyte.gz   emnist-letters-test-labels-idx1-ubyte.gz
cd tools/train-letters
EPOCHS=14 python3 train.py out
cp out/letters.bin out/letters.json ../../model/
```

訓練時會做大量資料增強（旋轉、縮放、傾斜、平移、筆畫變粗 / 變細、模糊），讓模型能應付手機拍到的筆跡。

網路架構：`conv32 → conv32 → pool → conv64 → conv64 → pool → conv128 → pool → dense128 → dense26`（ReLU），推論程式在 `js/recognizer.js`，純 JavaScript，不需要任何函式庫。
