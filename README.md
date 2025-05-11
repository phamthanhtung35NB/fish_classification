# Phân Loại Loài Cá Bằng Deep Learning 🐟

## 📌 Giới thiệu

Dự án này nhằm phân loại các loài cá khác nhau từ hình ảnh bằng cách sử dụng Mạng Nơ-ron Tích Chập (CNN). Mô hình đạt độ chính xác **98.98%** trên tập dữ liệu kiểm thử và hỗ trợ phân loại **hơn 40 loài cá**.

## 📊 Dữ liệu

[Link dataset kaggle](https://www.kaggle.com/datasets/markdaniellampa/fish-dataset)

- **Số lượng lớp:** 31 loài cá
- **Tổng số mẫu (val):** 1760
- **Định dạng dữ liệu:** Thư mục ảnh được gắn nhãn
- **Kích thước ảnh đầu vào:** (ví dụ 224x224 RGB)

## 🧠 Mô hình

- Mô hình CNN được xây dựng bằng TensorFlow/Keras (chi tiết trong `trainBestGpu.ipynb`)
- Bao gồm:
  - Các lớp Convolutional và MaxPooling
  - Chuẩn hóa BatchNormalization
  - Dropout để giảm overfitting
  - Các lớp Dense fully connected

## 🏋️‍♂️ Huấn luyện

- **Số epoch tối đa:** 30
- **Tối ưu hóa:** Adam
- **Hàm mất mát:** Categorical Crossentropy
- **Augmentation:** Có (lật ngang, zoom, v.v.)
- **Tách tập validation:** Có sử dụng

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

Biểu đồ độ chính xác và mất mát trong quá trình huấn luyện:

![Accuracy & Loss](https://github.com/user-attachments/assets/835a2f62-e595-4b3f-8727-6f1777565745)

Evaluating: 100%|██████████| 110/110 [00:17<00:00,  6.36it/s]
Độ chính xác tổng thể: 0.9898

Ma trận nhầm lẫn trên tập kiểm thử:

![Confusion Matrix](https://github.com/user-attachments/assets/088db298-c7a2-48fe-9b98-c5932cb7d619)



**Báo cáo phân loại nổi bật:**
- Phần lớn các lớp đạt **F1-score ≥ 0.97**
- F1-score trung bình trọng số: **0.99**

| Class Name           | Precision | Recall | F1-Score | Support |
| -------------------- | --------- | ------ | -------- | ------- |
| Bangus               | 0.97      | 1.00   | 0.99     | 34      |
| Big Head Carp        | 1.00      | 0.98   | 0.99     | 43      |
| Black Spotted Barb   | 1.00      | 1.00   | 1.00     | 40      |
| Catfish              | 0.95      | 1.00   | 0.98     | 62      |
| Climbing Perch       | 0.97      | 0.97   | 0.97     | 30      |
| Fourfinger Threadfin | 1.00      | 1.00   | 1.00     | 38      |
| Freshwater Eel       | 1.00      | 1.00   | 1.00     | 55      |
| Glass Perchlet       | 1.00      | 0.99   | 0.99     | 77      |
| Goby                 | 0.99      | 1.00   | 1.00     | 124     |
| Gold Fish            | 1.00      | 1.00   | 1.00     | 41      |
| Gourami              | 1.00      | 1.00   | 1.00     | 63      |
| Grass Carp           | 0.99      | 1.00   | 0.99     | 238     |
| Green Spotted Puffer | 1.00      | 1.00   | 1.00     | 22      |
| Indian Carp          | 0.98      | 1.00   | 0.99     | 53      |
| Indo-Pacific Tarpon  | 0.93      | 1.00   | 0.96     | 39      |


| Class Name            | Precision | Recall | F1-Score | Support |
| --------------------- | --------- | ------ | -------- | ------- |
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


| Metric        | Value |
| ------------- | ----- |
| Accuracy      | 0.99  |
| Macro Average | 0.99  |
| Weighted Avg  | 0.99  |
| Total Samples | 1760  |


## 📁 Các tệp trong dự án

- `trainBestGpu.ipynb`: Notebook huấn luyện và đánh giá
- `README.md`: Tài liệu hướng dẫn (file này)

## 🚀 Cách chạy

```bash
git clone <link-repo>
cd fish-classification
jupyter notebook trainBestGpu.ipynb
```

Yêu cầu cài đặt các thư viện:

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

