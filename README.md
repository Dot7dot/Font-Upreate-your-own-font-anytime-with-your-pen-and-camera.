# Font Anytime

用筆和相機，隨時做出自己的字型。

用手機打開網頁，拍下你手寫的 A–Z，系統會自動把每個字從照片裡切出來、用專門辨識手寫字母的 AI 模型判斷是哪個字母，最後輸出成可以安裝使用的 `.otf` 字型檔。

## 功能

- **資料夾**：每個資料夾就是一套字型，可新增、重新命名、刪除（資料存在瀏覽器的 IndexedDB，不會上傳到任何伺服器）。
- **拍照製作**：資料夾頁面上方是相機（也可以從相簿選照片），下方是 A–Z 每個字母目前的偵測狀態。
- **一次拍很多字**：自動切割照片中的每個字，用在 14 萬多個真人手寫字母上訓練的神經網路辨識對應的字母，沒偵測到的字母先留空。辨識完全在手機上執行，不需上傳照片。
- **重複字處理**：同一張照片拍到重複的字母時，會跳出視窗讓你點選最喜歡的那一個保留（若該字母已經建立過，原本的也會列入選項）。
- **手動修正**：不太確定的字會放在「未辨識的字」區，點一下會列出最可能的 3 個字母讓你一鍵指定；已建立的字母也可以刪除或改成其他字母。
- **字型預覽**：隨時輸入文字預覽目前的字型效果。
- **紙張風格**：整個介面像一張張撕下來的紙片、膠帶貼著的拍立得，支援深色模式。
- **輸出字型**：A–Z 全部建立後即可下載 `.otf` 字型檔（大小寫都對應到你的手寫字）；若還有缺字，會提示「還有尚未建立的字體喔」並返回畫面，不進行輸出。

## 拍照小技巧

- 用深色筆在白紙上寫 **大寫** 字母，字和字之間留一點空隙。
- 光線充足、避免陰影，盡量讓紙張填滿畫面但不要切到字。

## 技術

純前端靜態網站，不需要建置步驟：

| 步驟 | 做法 |
| --- | --- |
| 切割 | 自適應二值化（Bradley）→ 8 連通元件 → 合併同一字的筆畫 → 依閱讀順序排序 |
| 辨識 | 在 [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset) 手寫字母上訓練的 CNN（int8 量化，純 JS 推論），前處理會把筆畫粗細調成和訓練資料一致；訓練腳本在 `tools/train-letters/` |
| 向量化 | 沿像素邊界描出輪廓 → Douglas–Peucker 簡化 → 二次曲線平滑 |
| 字型 | [opentype.js](https://github.com/opentypejs/opentype.js) 產生 OpenType (CFF) 字型 |
| 紙張質感 | SVG `feTurbulence` 紙紋 + `feDisplacementMap` 手撕毛邊；字體 Huninn（粉圓）與 Caveat |

```
index.html
css/style.css
js/app.js       介面、資料夾、相機、重複字處理、輸出
js/segment.js   照片切割
js/recognizer.js 手寫字母辨識（CNN）
model/          辨識模型權重
js/font.js      向量化與字型產生
js/storage.js   IndexedDB 儲存
```

## 部署到 GitHub Pages

1. 到 repo 的 **Settings → Pages**。
2. **Source** 選 **Deploy from a branch**，Branch 選 `main`、資料夾選 `/ (root)`，按 Save。
3. 幾分鐘後即可用 `https://<你的帳號>.github.io/<repo 名稱>/` 開啟。

> 手機瀏覽器只有在 HTTPS 網址下才能使用相機，GitHub Pages 預設就是 HTTPS。

## 本機開發

任何靜態伺服器都可以，例如：

```bash
python3 -m http.server 8000
# 打開 http://localhost:8000
```
