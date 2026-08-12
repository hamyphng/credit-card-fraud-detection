# Kế Hoạch Dự Án: Phát Hiện Gian Lận Thẻ Tín Dụng (Credit Card Fraud Detection)

Dự án này nhằm xây dựng một mô hình học máy (Machine Learning) có khả năng phát hiện các giao dịch gian lận thẻ tín dụng dựa trên lịch sử giao dịch. Đây là một bài toán phân loại nhị phân (Binary Classification) với đặc thù **dữ liệu cực kỳ mất cân bằng** (Imbalanced Data), trong đó số lượng giao dịch gian lận (Fraud) chiếm tỷ lệ rất nhỏ (< 0.2%).

---

## 1. Cơ Cấu Thư Mục Dự Án

Thư mục dự án được tổ chức như sau để đảm bảo tính chuyên nghiệp, dễ quản lý và sẵn sàng đưa vào sản xuất (production-ready):

```text
finance/
├── .vscode/
│   └── settings.json           # Cấu hình môi trường ảo cho VS Code
├── data/
│   └── creditcard.csv          # File dữ liệu gốc (đã có sẵn)
├── notebooks/                  # Nơi lưu trữ các file Jupyter Notebook thử nghiệm
│   ├── 01_eda.ipynb            # Phân tích khám phá dữ liệu (EDA)
│   ├── 02_preprocessing.ipynb  # Thử nghiệm tiền xử lý & Xử lý mất cân bằng
│   ├── 03_modeling.ipynb       # Huấn luyện các mô hình baseline & nâng cao
│   └── 04_evaluation.ipynb     # Đánh giá chi tiết & tối ưu hóa ngưỡng quyết định
├── src/                        # Mã nguồn Python tái sử dụng dạng module
│   ├── __init__.py
│   ├── data_loader.py          # Module load và lưu trữ dữ liệu
│   ├── preprocessing.py        # Module chuẩn hóa, scale và split dữ liệu
│   ├── model.py                # Module định nghĩa cấu trúc và huấn luyện mô hình
│   └── evaluation.py           # Module tính toán các metrics (Precision, Recall, AUPRC,...)
├── tests/                      # Thư mục chứa mã nguồn kiểm thử (Unit Tests)
│   ├── __init__.py
│   ├── test_data.py
│   └── test_model.py
├── requirements.txt            # Danh sách thư viện cần thiết
└── plan.md                     # File kế hoạch dự án này
```

---

## 2. Kế Hoạch Từng Bước & Hướng Dẫn Thực Hiện

### Bước 1: Phân Tích Khám Phá Dữ Liệu (EDA)
* **Mục tiêu**: Hiểu cấu trúc dữ liệu, phân phối của các thuộc tính và mức độ mất cân bằng lớp.
* **Nơi thực hiện**: `notebooks/01_eda.ipynb`
* **Công cụ**: `pandas`, `numpy`, `matplotlib`, `seaborn`.
* **Nhiệm vụ cụ thể**:
  1. Đọc dữ liệu từ `data/creditcard.csv`.
  2. Kiểm tra số lượng dòng, cột, dữ liệu khuyết thiếu (missing values).
  3. Thống kê tỷ lệ phân phối giữa lớp `Class = 0` (Giao dịch thường) và `Class = 1` (Gian lận).
  4. Trực quan hóa phân phối của trường `Amount` (Số tiền) và `Time` (Thời gian) để nhận diện các đặc trưng phân phối lệch (skewed distribution).
  5. Xem xét phân phối của các biến ẩn `V1` đến `V28` (kết quả của phép biến đổi PCA).
  6. Sử dụng heatmap để trực quan hóa ma trận tương quan (correlation matrix) giữa các thuộc tính.

### Bước 2: Tiền Xử Lý Dữ Liệu (Preprocessing)
* **Mục tiêu**: Chuẩn bị dữ liệu sạch và chuẩn hóa trước khi đưa vào mô hình học máy.
* **Nơi thực hiện**: `notebooks/02_preprocessing.ipynb` và module `src/preprocessing.py`.
* **Công cụ**: `scikit-learn` (`RobustScaler`, `StandardScaler`, `train_test_split`).
* **Nhiệm vụ cụ thể**:
  1. Chuẩn hóa (Scale) các trường `Amount` và `Time` (Vì `V1-V28` đã được scale qua PCA, còn `Amount` và `Time` có khoảng giá trị rất khác biệt). Nên sử dụng `RobustScaler` để giảm thiểu ảnh hưởng của các điểm dị biệt (outliers).
  2. Chia dữ liệu thành tập **Train / Validation / Test** (Ví dụ tỷ lệ: 70% / 15% / 15% hoặc 80% / 20% Train/Test).
     * **LƯU Ý CỰC KỲ QUAN TRỌNG**: Phải sử dụng `stratify=y` trong `train_test_split` để giữ nguyên tỷ lệ phân phối mất cân bằng của hai lớp trên cả 3 tập dữ liệu.

### Bước 3: Xử Lý Mất Cân Bằng Lớp (Handling Imbalance)
* **Mục tiêu**: Giúp mô hình không bị "thiên vị" (bias) nghiêng hoàn toàn về lớp đa số (giao dịch thường).
* **Nơi thực hiện**: `notebooks/02_preprocessing.ipynb`
* **Công cụ**: `imbalanced-learn` (`SMOTE`, `RandomUnderSampler`, `SMOTEENN`).
* **Nhiệm vụ cụ thể**:
  1. Thử nghiệm các phương pháp:
     * **Oversampling (Tăng mẫu thiểu số)**: Sử dụng thuật toán SMOTE để sinh ra các mẫu giao dịch gian lận nhân tạo.
     * **Undersampling (Giảm mẫu đa số)**: Sử dụng RandomUnderSampler để giảm số lượng giao dịch thường ngang bằng với giao dịch gian lận.
     * **Cost-Sensitive Learning (Học nhạy cảm chi phí)**: Sử dụng thuộc tính `class_weight='balanced'` có sẵn trong các mô hình của `scikit-learn` thay vì resample dữ liệu trực tiếp.
  2. **QUY TẮC VÀNG**: Chỉ thực hiện Oversampling/Undersampling trên **tập Train**. Tuyệt đối không resample trên tập Validation và Test để tránh rò rỉ dữ liệu (Data Leakage) dẫn đến đánh giá sai lệch hiệu năng của mô hình.

### Bước 4: Xây Dựng & Huấn Luyện Mô Hình (Modeling)
* **Mục tiêu**: Huấn luyện và tìm ra thuật toán tối ưu nhất cho bài toán.
* **Nơi thực hiện**: `notebooks/03_modeling.ipynb` và module `src/model.py`.
* **Công cụ**: `scikit-learn` (Logistic Regression, Random Forest), `xgboost`, `lightgbm`.
* **Nhiệm vụ cụ thể**:
  1. Xây dựng mô hình Baseline đơn giản bằng **Logistic Regression** hoặc **Decision Tree** để làm mốc so sánh.
  2. Thử nghiệm các mô hình phức tạp hơn:
     * **Random Forest**: Hiệu quả với tập dữ liệu dạng bảng và có khả năng chống overfitting khá tốt.
     * **XGBoost & LightGBM**: Các thuật toán Boosting hàng đầu hiện nay, tối ưu tốt cho các bài toán phân loại mất cân bằng và chạy rất nhanh.
  3. Sử dụng K-Fold Cross-Validation (cụ thể là `StratifiedKFold`) trên tập Train để đánh giá độ ổn định của các mô hình.

### Bước 5: Đánh Giá Chi Tiết & Tối Ưu Hóa Ngưỡng (Evaluation & Tuning)
* **Mục tiêu**: Đánh giá chính xác mô hình dựa trên các chỉ số phù hợp với dữ liệu mất cân bằng và tìm ngưỡng quyết định (threshold) tối ưu nhất.
* **Nơi thực hiện**: `notebooks/04_evaluation.ipynb` và module `src/evaluation.py`.
* **Công cụ**: `scikit-learn` (`classification_report`, `precision_recall_curve`, `confusion_matrix`).
* **Nhiệm vụ cụ thể**:
  1. **LỰA CHỌN METRIC**:
     * **KHÔNG** dùng chỉ số Accuracy (Độ chính xác toàn cục) vì nếu mô hình đoán toàn bộ là giao dịch thường thì accuracy vẫn đạt 99.8%.
     * Dùng **Recall** (Sensitivity): Tỷ lệ phát hiện được bao nhiêu phần trăm trong tổng số giao dịch gian lận thực tế (quan trọng nhất vì bỏ sót gian lận sẽ gây thất thoát tiền bạc lớn).
     * Dùng **Precision**: Tỷ lệ giao dịch được mô hình cảnh báo gian lận thực sự là gian lận (quan trọng để tránh làm phiền khách hàng với các cảnh báo sai).
     * Dùng **F1-Score**: Trung bình điều hòa giữa Precision và Recall.
     * Dùng **AUPRC** (Area Under the Precision-Recall Curve): Đường cong đánh giá toàn diện cho bài toán mất cân bằng (tốt hơn nhiều so với ROC-AUC).
  2. **TỐI ƯU HÓA NGƯỠNG QUYẾT ĐỊNH (Threshold Tuning)**:
     * Ngưỡng mặc định của mô hình phân loại là `0.5`. Bạn cần vẽ đường cong Precision-Recall theo sự thay đổi của threshold từ `0` đến `1`.
     * Tìm điểm ngưỡng tối ưu nhất sao cho tối đa hóa Recall nhưng vẫn giữ Precision ở mức chấp nhận được (ví dụ: tối đa hóa F1-Score hoặc định nghĩa một hàm mất mát chi phí: mỗi False Negative tốn $100, mỗi False Positive tốn $5).

### Bước 6: Đóng Gói Mã Nguồn (Refactoring) & Kiểm Thử (Testing)
* **Mục tiêu**: Chuyển đổi code thử nghiệm từ file Notebook (.ipynb) thành các module Python (.py) sạch sẽ, có cấu trúc tái sử dụng và viết Unit Tests.
* **Nơi thực hiện**: Thư mục `src/` và `tests/`.
* **Công cụ**: `unittest` hoặc `pytest`.
* **Nhiệm vụ cụ thể**:
  1. Chuyển các hàm load dữ liệu, preprocess dữ liệu sang `src/data_loader.py` và `src/preprocessing.py`.
  2. Viết class huấn luyện mô hình và lưu mô hình (`.pkl` hoặc `.model`) sang `src/model.py`.
  3. Viết Unit Tests kiểm tra tính đúng đắn của việc chuẩn hóa dữ liệu hoặc kiểm tra xem kích thước đầu ra của hàm chia tập dữ liệu có chính xác không trong thư mục `tests/`.

---

## 3. Cách Sử Dụng Workspace Này Hiệu Quả
1. **Sử dụng môi trường ảo**: Hãy đảm bảo bạn đã chọn Python Interpreter trỏ tới `.venv/Scripts/python.exe` trong VS Code (Nhấn `Ctrl + Shift + P` -> gõ `Python: Select Interpreter` -> chọn `.venv`).
2. **Làm việc với Jupyter Notebooks**: Khi mở các file notebook trong thư mục `notebooks/`, hãy chọn Kernel là `.venv` ở góc trên cùng bên phải của VS Code.
3. **Thực hiện theo thứ tự**: Nên bắt đầu làm từ file `01_eda.ipynb` để có cái nhìn trực quan về dữ liệu trước khi chuyển sang các bước tiếp theo.
