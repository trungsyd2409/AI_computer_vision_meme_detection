# Meme Reaction Cam

Webcam nhận diện **biểu cảm khuôn mặt + cử chỉ tay** của bạn, rồi hiện ảnh meme giống nhất ở bên phải.
Nếu không đủ giống meme nào thì bảng bên phải để trống ("no match").

```
webcam ─► MediaPipe (Face + Hand Landmarker) ─► vector 198 số ─► classifier (scikit-learn)
       ─► làm mượt + ngưỡng (stabilizer) ─► hiện meme / không hiện gì
```

## 1. Cài đặt (Windows)

Cần Python **3.10 – 3.12** (khuyên dùng 3.11) và webcam.

```powershell
cd F:\Chunwan\SourceCode\AI\AI_computer_vision_meme_detection
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Lần chạy đầu tiên, code tự tải 2 model MediaPipe (`face_landmarker.task`, `hand_landmarker.task`) vào `models/`.

## 2. Thêm ảnh meme

Bỏ ảnh vào thư mục `memes/`. **Tên file = tên nhãn (label)**, ví dụ:

```
memes/
  hamster_peace.jpg
  shocked_cat.png
  thinking_guy.webp
```

Hỗ trợ: jpg, jpeg, png, webp, bmp, gif (lấy frame đầu). Nên đặt tên không dấu, không khoảng trắng.

## 3. Thu dữ liệu – `collect.py`

```powershell
python collect.py
```

Cửa sổ hiện webcam bên trái, meme cần bắt chước bên phải.

| Phím | Tác dụng |
|---|---|
| `SPACE` | đếm ngược 3s rồi ghi 4s (mỗi frame = 1 mẫu) |
| `N` / `P` | meme tiếp theo / trước |
| `U` | xóa lần ghi vừa rồi |
| `Q` / `ESC` | thoát |

Mẹo để model tốt:

- Mỗi meme ghi **3–4 lần** (session), mỗi lần hơi đổi khoảng cách, góc mặt, ánh sáng, tay trái/phải.
- **Bắt buộc nên ghi nhãn `_none`** (mục cuối cùng): mặt bình thường, nói chuyện, nhìn quanh, cử động tay lung tung. Đây là cách để máy học "không giống meme nào".
- Ghi thêm cho 1 meme: `python collect.py --label hamster_peace`

Dữ liệu lưu ở `data/dataset.csv` (cột `label`, `session`, và 198 feature) – mở bằng pandas để khám phá được.

## 4. Train – `train.py`

```powershell
python train.py
```

Script so sánh 5 model (Logistic Regression, SVM, kNN, Random Forest, MLP) bằng cross-validation, in bảng F1 / accuracy,
classification report và confusion matrix, rồi lưu model tốt nhất vào `models/classifier.joblib`.

Cross-validation chia theo **session** (`StratifiedGroupKFold`): các frame trong cùng 1 lần ghi gần như giống hệt nhau,
nếu chia ngẫu nhiên theo frame thì điểm sẽ cao ảo (data leakage).

Chọn 1 model cụ thể: `python train.py --model svm`

## 5. Chạy app – `app.py`

```powershell
python app.py
python app.py --threshold 0.8 --camera 1
```

| Phím | Tác dụng |
|---|---|
| `+` / `-` | tăng / giảm ngưỡng |
| `[` / `]` | giảm / tăng số frame cần giữ trước khi hiện meme |
| `D` | bật / tắt chữ debug |
| `S` | chụp màn hình vào `screenshots/` |
| `Q` / `ESC` | thoát |

**Cửa sổ cố định:** kích thước cửa sổ được tính 1 lần từ frame đầu tiên và không kéo giãn được, nên hình camera
luôn đúng tỷ lệ (chỉ thu nhỏ đều 2 chiều). Cửa sổ tự thu nhỏ nếu màn hình nhỏ.
Đổi kích thước: `python app.py --height 480`. Khi chạy, dòng `Camera 0: 1280x720 (aspect 1.78)` cho biết độ phân giải thật;
nếu hình vẫn trông méo, webcam có thể không hỗ trợ 16:9 → thử `--cam-width 640 --cam-height 480`.

**Cách quyết định có hiện meme không** (`meme_cam/stabilizer.py`) – "hiện nhanh, tắt chậm":

1. Làm mượt xác suất bằng EMA (`alpha = 0.6`, 1.0 = không làm mượt).
2. Meme hiện khi nhãn cao nhất **không phải `_none`**, xác suất ≥ ngưỡng (0.70) trong **2 frame liên tiếp** (~80 ms ở 25 fps).
3. Meme chỉ tắt khi xác suất < ngưỡng − 0.10 trong **4 frame liên tiếp** → 1 frame nhận diện sai không làm meme nhấp nháy.
4. Mất mặt/tay quá 4 frame → không hiện gì.

**Giảm độ trễ (delay):**

- Camera được đọc trong 1 thread riêng và chỉ giữ **frame mới nhất** (bỏ frame cũ trong hàng đợi của OpenCV).
- MediaPipe chạy trên ảnh thu nhỏ 640px (`--process-width`), nhanh hơn ~2–3 lần mà landmark gần như không đổi.
- Debug hiển thị `fps` và `mediapipe: xx ms` để biết đang chậm ở đâu.
- Muốn tức thì nhất: `python app.py --hold 1 --alpha 1.0` (có thể nhấp nháy hơn). Khi chạy bấm `[` / `]` để giảm / tăng số frame giữ.

Chỉnh mặc định trong `meme_cam/config.py`.

## Feature (198 chiều) – `meme_cam/features.py`

| Nhóm | Số chiều | Ý nghĩa |
|---|---|---|
| face_present | 1 | có mặt hay không |
| blendshapes | 52 | điểm biểu cảm 0–1 của MediaPipe (smile, jawOpen, eyeBlink, browInnerUp…) |
| head pose | 3 | pitch / yaw / roll của đầu |
| mỗi tay (Left, Right) × 71 | 142 | có tay; 21 điểm khớp so với cổ tay (không phụ thuộc kích thước/vị trí); độ duỗi 5 ngón; vị trí tay so với mặt |

## Cấu trúc

```
app.py            chạy camera meme
collect.py        thu dữ liệu
train.py          train + đánh giá model
meme_cam/
  config.py       đường dẫn + tham số mặc định
  features.py     MediaPipe -> vector feature
  stabilizer.py   làm mượt + ngưỡng + hysteresis
  ui.py           vẽ khung mặt, tay, chữ debug, ghép 2 panel
  camera.py       mở webcam (DirectShow trên Windows)
  memes.py        đọc ảnh meme
tests/            pytest cho features + stabilizer
```

Chạy test: `python -m pytest -q`

## Train nhiều dữ liệu hơn có làm app chậm/nặng hơn không?

Phần lớn thời gian mỗi frame là MediaPipe (~20–40 ms), và phần này **không đổi** dù dataset lớn hay nhỏ.
Classifier chỉ tốn khoảng 0.2–2 ms/frame:

| Model | Thêm dữ liệu → dự đoán chậm hơn? | File model to hơn? |
|---|---|---|
| logreg, mlp | không | không |
| svm | tăng nhẹ (theo số support vector) | tăng nhẹ |
| knn | có (so với mọi mẫu) | có, tăng tuyến tính |
| random_forest | gần như không (~15 ms) | tăng nhẹ |

Thêm **meme** (nhãn) thì chậm hơn không đáng kể. Chỉ có `train.py` chạy lâu hơn khi data nhiều.

## Lỗi thường gặp

- **Không mở được camera** → thử `--camera 1`, đóng Zoom/Teams đang dùng camera.
- **Tải model lỗi** → tải tay 2 file trong `meme_cam/config.py` (`FACE_MODEL_URL`, `HAND_MODEL_URL`) bỏ vào `models/`.
- **`module 'mediapipe' has no attribute 'solutions'`** → code này dùng MediaPipe **Tasks API** mới, không dùng `mp.solutions` (đã bị bỏ ở bản mới).
- **Meme hay hiện sai** → ghi thêm `_none`, tăng ngưỡng (`+`), ghi thêm session cho meme bị nhầm (xem confusion matrix).
- **Không bao giờ hiện meme** → giảm ngưỡng (`-`), kiểm tra debug xem xác suất đang bao nhiêu.
- Sau khi đổi code trong `features.py` → xóa `data/dataset.csv` và ghi lại (feature đã khác).
