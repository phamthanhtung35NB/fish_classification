
# Phân Loại Loài Cá Bằng Deep Learning 🐟

## 📌 Giới thiệu

Dự án này nhằm phân loại các loài cá khác nhau từ hình ảnh bằng cách sử dụng Mạng Nơ-ron Tích Chập (CNN). Mô hình đạt độ chính xác **98.98%** trên tập dữ liệu kiểm thử và hỗ trợ phân loại **31 loài cá**.

## 📊 Dữ liệu

[Link dataset Kaggle](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset)

*Thuộc tính của tập dữ liệu:*
* **Số lượng lớp:** 31 loài cá
* **Tổng số hình ảnh:** 13.304 ảnh
* **Số lượng ảnh trong tập huấn luyện:** 8.791 ảnh
* **Số lượng ảnh trong tập kiểm tra:** 2.751 ảnh
* **Số lượng lớp (nhãn):** 1.760 lớp
* **Định dạng dữ liệu:** Thư mục ảnh được gắn nhãn
* **Kích thước ảnh đầu vào:** 224x224 RGB

## 🧠 Mô hình

* Sử dụng mô hình **ResNet** (transfer learning) với PyTorch
* Bao gồm:

  * Tiền xử lý ảnh bằng `torchvision.transforms`
  * Fine-tuning ResNet với trọng số pretrained
  * Chuẩn hóa ảnh, augmentation và resize về kích thước chuẩn

## 🏋️‍♂️ Huấn luyện

* **Epoch:** tối đa 30
* **Tối ưu hóa:** AdamW
* **Hàm mất mát:** CrossEntropyLoss
* **Augmentation:** Resize, Horizontal Flip, Rotation
* **Tách validation:** sử dụng `torch.utils.data.random_split`

## 📈 Kết quả

* **Độ chính xác trên tập test:** 98.98%
* **Confusion matrix & classification report:** trực quan bằng `seaborn` và `sklearn`


| Tên tiếng Anh          | Tên tiếng Việt         |    | Tên tiếng Anh          | Tên tiếng Việt         |
|------------------------|------------------------|----|------------------------|------------------------|
| Bangus                | Cá măng                |    | Mullet                | Cá đối                |
| Big Head Carp         | Cá mè hoa              |    | Pangasius             | Cá tra                |
| Black Spotted Barb    | Cá chép đốm đen        |    | Perch                 | Cá rô biển            |
| Catfish               | Cá trê                 |    | Scat Fish             | Cá hồng két chấm       |
| Climbing Perch        | Cá rô đồng             |    | Silver Barb           | Cá mè trắng            |
| Fourfinger Threadfin  | Cá chỉ vàng            |    | Silver Carp           | Cá mè trắng lớn        |
| Freshwater Eel        | Lươn nước ngọt         |    | Silver Perch          | Cá vền bạc             |
| Glass Perchlet        | Cá kính nhỏ            |    | Snakehead             | Cá lóc                 |
| Goby                  | Cá bống                |    | Tenpounder            | Cá cháo bạc            |
| Gold Fish             | Cá vàng                |    | Tilapia               | Cá rô phi              |
| Gourami               | Cá tai tượng           |    | Indian Carp           | Cá trắm Ấn Độ          |
| Grass Carp            | Cá trắm cỏ             |    | Indo-Pacific Tarpon   | Cá cháo                |
| Green Spotted Puffer  | Cá nóc xanh chấm       |    | Jaguar Gapote         | Cá rồng đốm            |
| Janitor Fish          | Cá lau kiếng           |    | Knifefish             | Cá dao                 |
| Long-Snouted Pipefish | Cá ngựa mõm dài        |    | Mosquito Fish         | Cá diệt lăng quăng     |
| Mudfish               | Cá lóc bùn             |    |                        |                        |


## 📈 Hiệu suất

- **Độ chính xác:**
  - Train: ~99.8%
  - Validation: ~98.9%
- **Hàm mất mát:**
  - Train: ~0.01
  - Validation: ~0.05


## Console print

Epoch 22/30 [Train]: 100%|██████████| 550/550 [01:30<00:00,  6.08it/s, acc=1.0000, loss=0.0020, lr=0.000020]
  
Epoch 22/30 [Val]: 100%|██████████| 110/110 [00:26<00:00,  4.16it/s, acc=0.9375, loss=0.2573]

Epoch 22/30 - Train Loss: 0.0135, Train Acc: 0.9965, Val Loss: 0.0438, Val Acc: 0.9909

Đã lưu mô hình tốt nhất với độ chính xác: 0.9909

----

=== GPU Usage Information ===

GPU: NVIDIA GeForce RTX 3050 Laptop GPU

Memory Allocated: 0.26 GB

Max Memory Allocated: 0.66 GB

Memory Reserved: 0.82 GB

Total Memory: 4.00 GB
Memory Utilization: 6.44%

Epoch 27/30 [Train]: 100%|██████████| 550/550 [01:29<00:00,  6.17it/s, acc=0.8889, loss=0.1834, lr=0.000020]

Epoch 27/30 [Val]: 100%|██████████| 110/110 [00:25<00:00,  4.30it/s, acc=0.9375, loss=0.2455]

Epoch 00027: reducing learning rate of group 0 to 4.0000e-06.

Epoch 27/30 - Train Loss: 0.0135, Train Acc: 0.9960, Val Loss: 0.0555, Val Acc: 0.9898

Early stopping tại epoch 27

Biểu đồ độ chính xác và mất mát trong quá trình huấn luyện:

![Accuracy & Loss](https://github.com/user-attachments/assets/835a2f62-e595-4b3f-8727-6f1777565745)

Evaluating: 100%|██████████| 110/110 [00:17<00:00,  6.36it/s]
Độ chính xác tổng thể: 0.9898

Ma trận nhầm lẫn trên tập kiểm thử:

![Confusion Matrix](https://github.com/user-attachments/assets/088db298-c7a2-48fe-9b98-c5932cb7d619)



**Báo cáo phân loại nổi bật:**
- Phần lớn các lớp đạt **F1-score ≥ 0.98**
- F1-score trung bình trọng số: **0.99**

| Class Name            | Precision | Recall | F1-Score | Support |
| --------------------- | --------- | ------ | -------- | ------- |
| Bangus                | 0.97      | 1.00   | 0.99     | 34      |
| Big Head Carp         | 1.00      | 0.98   | 0.99     | 43      |
| Black Spotted Barb    | 1.00      | 1.00   | 1.00     | 40      |
| Catfish               | 0.95      | 1.00   | 0.98     | 62      |
| Climbing Perch        | 0.97      | 0.97   | 0.97     | 30      |
| Fourfinger Threadfin  | 1.00      | 1.00   | 1.00     | 38      |
| Freshwater Eel        | 1.00      | 1.00   | 1.00     | 55      |
| Glass Perchlet        | 1.00      | 0.99   | 0.99     | 77      |
| Goby                  | 0.99      | 1.00   | 1.00     | 124     |
| Gold Fish             | 1.00      | 1.00   | 1.00     | 41      |
| Gourami               | 1.00      | 1.00   | 1.00     | 63      |
| Grass Carp            | 0.99      | 1.00   | 0.99     | 238     |
| Green Spotted Puffer  | 1.00      | 1.00   | 1.00     | 22      |
| Indian Carp           | 0.98      | 1.00   | 0.99     | 53      |
| Indo-Pacific Tarpon   | 0.93      | 1.00   | 0.96     | 39      |
| Jaguar Gapote         | 1.00      | 1.00   | 1.00     | 44      |
| Janitor Fish          | 0.97      | 1.00   | 0.98     | 58      |
| Knifefish             | 1.00      | 1.00   | 1.00     | 65      |
| Long-Snouted Pipefish | 1.00      | 1.00   | 1.00     | 52      |
| Mosquito Fish         | 1.00      | 1.00   | 1.00     | 51      |
| Mudfish               | 1.00      | 0.88   | 0.94     | 34      |
| Mullet                | 0.97      | 0.97   | 0.97     | 38      |
| Pangasius             | 1.00      | 1.00   | 1.00     | 38      |
| Perch                 | 1.00      | 1.00   | 1.00     | 60      |
| Scat Fish             | 1.00      | 1.00   | 1.00     | 33      |
| Silver Barb           | 1.00      | 0.98   | 0.99     | 64      |
| Silver Carp           | 1.00      | 1.00   | 1.00     | 48      |
| Silver Perch          | 0.98      | 1.00   | 0.99     | 57      |
| Snakehead             | 0.96      | 0.96   | 0.96     | 47      |
| Tenpounder            | 1.00      | 0.91   | 0.95     | 56      |
| Tilapia               | 1.00      | 0.98   | 0.99     | 56      |


| Class Name            | Precision | Recall | F1-Score | Support |
| --------------------- | --------- | ------ | -------- | ------- |
|   accuracy            |           |        |   0.99   |   1760  |
|           macro avg   |   0.99    |  0.99  |   0.99   |   1760  |
|       weighted avg    |   0.99    |  0.99  |   0.99   |   1760  |

| Metric        | Value |
| ------------- | ----- |
| Accuracy      | 0.99  |
| Macro Average | 0.99  |
| Weighted Avg  | 0.99  |
| Total Samples | 1760  |


## Thử mô hình:
```
Tìm thấy 24 file ảnh
  ✓ Ảnh: Bangus 19.jpg, Thực tế: Bangus, Dự đoán: Bangus, Độ tin cậy: 1.0000
  ✓ Ảnh: Bangus 21.jpg, Thực tế: Bangus, Dự đoán: Bangus, Độ tin cậy: 0.9998
  ✓ Ảnh: Bangus 32.jpg, Thực tế: Bangus, Dự đoán: Bangus, Độ tin cậy: 0.9999
  ✓ Ảnh: Bangus 35.jpg, Thực tế: Bangus, Dự đoán: Bangus, Độ tin cậy: 1.0000
  ✓ Ảnh: Bangus 42.jpg, Thực tế: Bangus, Dự đoán: Bangus, Độ tin cậy: 1.0000
  ✓ Ảnh: Black Spotted Barb 15.jpg, Thực tế: Black Spotted Barb, Dự đoán: Black Spotted Barb, Độ tin cậy: 1.0000
  ✓ Ảnh: Black Spotted Barb 24.jpg, Thực tế: Black Spotted Barb, Dự đoán: Black Spotted Barb, Độ tin cậy: 0.9997
  ✓ Ảnh: Black Spotted Barb 33.jpg, Thực tế: Black Spotted Barb, Dự đoán: Black Spotted Barb, Độ tin cậy: 0.9997
  ✓ Ảnh: Black Spotted Barb 34.jpg, Thực tế: Black Spotted Barb, Dự đoán: Black Spotted Barb, Độ tin cậy: 0.9999
  ✓ Ảnh: Black Spotted Barb 8.jpg, Thực tế: Black Spotted Barb, Dự đoán: Black Spotted Barb, Độ tin cậy: 0.9997
  ✓ Ảnh: Gold Fish 22.jpg, Thực tế: Gold Fish, Dự đoán: Gold Fish, Độ tin cậy: 0.9992
  ✓ Ảnh: Gold Fish 24.jpg, Thực tế: Gold Fish, Dự đoán: Gold Fish, Độ tin cậy: 1.0000
  ✓ Ảnh: Gold Fish 33.jpg, Thực tế: Gold Fish, Dự đoán: Gold Fish, Độ tin cậy: 0.9999
  ✓ Ảnh: Gold Fish 40.jpg, Thực tế: Gold Fish, Dự đoán: Gold Fish, Độ tin cậy: 0.9946
  ✓ Ảnh: Gold Fish 43.jpg, Thực tế: Gold Fish, Dự đoán: Gold Fish, Độ tin cậy: 0.9999
  ✓ Ảnh: Indo-Pacific Tarpon 13.jpg, Thực tế: Indo-Pacific Tarpon, Dự đoán: Indo-Pacific Tarpon, Độ tin cậy: 1.0000
  ✓ Ảnh: Indo-Pacific Tarpon 16.jpg, Thực tế: Indo-Pacific Tarpon, Dự đoán: Indo-Pacific Tarpon, Độ tin cậy: 1.0000
  ✓ Ảnh: Indo-Pacific Tarpon 22.jpg, Thực tế: Indo-Pacific Tarpon, Dự đoán: Indo-Pacific Tarpon, Độ tin cậy: 1.0000
  ✓ Ảnh: Indo-Pacific Tarpon 31.jpg, Thực tế: Indo-Pacific Tarpon, Dự đoán: Indo-Pacific Tarpon, Độ tin cậy: 1.0000
  ✓ Ảnh: Indo-Pacific Tarpon 9.jpg, Thực tế: Indo-Pacific Tarpon, Dự đoán: Indo-Pacific Tarpon, Độ tin cậy: 0.9998
  ✓ Ảnh: Tenpounder 061.jpg, Thực tế: Tenpounder, Dự đoán: Tenpounder, Độ tin cậy: 0.9999
  ✓ Ảnh: Tenpounder 062.jpg, Thực tế: Tenpounder, Dự đoán: Tenpounder, Độ tin cậy: 0.9993
  ✓ Ảnh: Tenpounder 067.jpg, Thực tế: Tenpounder, Dự đoán: Tenpounder, Độ tin cậy: 0.9991
  ✓ Ảnh: Tenpounder 069.jpg, Thực tế: Tenpounder, Dự đoán: Tenpounder, Độ tin cậy: 0.9930

Kết quả kiểm tra:
Tổng số ảnh: 24
Dự đoán đúng: 24
Độ chính xác: 1.0000
```

![Image](https://github.com/user-attachments/assets/af26d836-951e-4fad-be65-acbbd9efb36c)


![Image](https://github.com/user-attachments/assets/d7371736-0dbb-438f-9c42-5587d4d3e40d)


#### Ảnh thực tế test ngoài tập dữ liệu:

![Image](https://github.com/user-attachments/assets/71532ce6-2a41-4c68-80fd-7fe0b531d77a)


## 📁 Các tệp trong dự án

- `trainBestGpu.ipynb`: Notebook huấn luyện và đánh giá
- `README.md`: Tài liệu hướng dẫn (file này)

## 🚀 Cách chạy

```bash
git clone https://github.com/phamthanhtung35NB/fish_classification.git
cd fish-classification
jupyter notebook trainBestGpu.ipynb
```

Yêu cầu cài đặt các thư viện:

```bash
pip install torch torchvision matplotlib seaborn scikit-learn tqdm pillow
```

