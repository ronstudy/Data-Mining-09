# Khai phá Dữ liệu - Phân loại và Phân cụm Đậu khô (Dry Bean Dataset)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipeline-F7931E?logo=scikit-learn&logoColor=white)
![Pandas & NumPy](https://img.shields.io/badge/Pandas%20%26%20NumPy-Data%20Processing-150458?logo=pandas&logoColor=white)
![Matplotlib & Seaborn](https://img.shields.io/badge/Matplotlib%20%26%20Seaborn-300%20DPI%20Charts-11557C)
![Code Style](https://img.shields.io/badge/PEP%208-%E2%89%A4%2088%20chars-4B8BBE)
![Reproducibility](https://img.shields.io/badge/RANDOM__STATE-42%20(100%25%20Deterministic)-2EA44F)
![Status](https://img.shields.io/badge/Ti%E1%BA%BFn%20%C4%91%E1%BB%99-16%2F16%20Tasks%20(100%25)-success)

> **Tóm tắt mục tiêu:** Xây dựng quy trình khai phá dữ liệu (*Data Mining Pipeline*) hoàn chỉnh và có tính tái lập 100% trong một tệp mã nguồn duy nhất [`main.py`](main.py) trên tập dữ liệu *Dry Bean Dataset* (13.611 mẫu gốc, 16 đặc trưng hình thái, 7 giống đậu). Quy trình đi xuyên suốt từ kiểm tra tính toàn vẹn và tiền xử lý dữ liệu (kiểm tra khuyết thiếu, loại bỏ trùng lặp, phát hiện ngoại lai bằng IQR và giữ nguyên biến thiên sinh học tự nhiên), phân tích khám phá dữ liệu (EDA đơn biến và đa biến), đến giải quyết hai bài toán nghiên cứu cốt lõi bằng học có giám sát (`GaussianNB`, `DecisionTreeClassifier`) và học không giám sát (`KMeans`, `PCA`).

---

## 1. Kiến trúc & Cấu trúc dự án

### 1.1. Cấu trúc cây thư mục

```text
Data-Mining-09/
├── dry_beans.csv                 # Dữ liệu gốc Dry Bean Dataset (13.611 dòng × 17 cột)
├── main.py                       # Script thực thi duy nhất chạy trọn vẹn toàn bộ pipeline (PEP 8)
├── README.md                     # Tài liệu tổng quan dự án, kết quả thực nghiệm và hướng dẫn chạy
├── outputs/                      # Thư mục ảnh đầu ra tự động tạo, chứa 12 biểu đồ PNG chuẩn 300 DPI
│   ├── 01_phan_bo_lop.png
│   ├── 02_histogram_dac_trung.png
│   ├── 03_boxplot_ngoai_lai.png
│   ├── 04_ma_tran_tuong_quan.png
│   ├── 05_boxplot_theo_lop.png
│   ├── 06_confusion_matrix.png
│   ├── 07_so_sanh_mo_hinh.png
│   ├── 08_feature_importances.png
│   ├── 09_elbow_silhouette.png
│   ├── 10_crosstab_cum_lop.png
│   ├── 11_pca_cum.png
│   └── 12_pca_lop_that.png
└── Planing/                      # Bộ 6 tài liệu đặc tả, thiết kế và quản lý tiến độ dự án
    ├── PRD.md                    # Tài liệu yêu cầu sản phẩm, câu hỏi nghiên cứu & tiêu chí nghiệm thu
    ├── ARCHITECTURE.md           # Kiến trúc kỹ thuật, luồng dữ liệu & quy ước xử lý giữa các bước
    ├── DESIGN.md                 # Tiêu chuẩn trực quan hóa (style, font, bảng màu tab10, danh mục 12 biểu đồ)
    ├── RULES.md                  # Quy chuẩn PEP 8, xử lý ngoại lệ & danh sách điều cấm tuyệt đối
    ├── TASKS.md                  # Danh sách 16 task chia theo 4 sprint và tiêu chí kiểm tra từng task
    └── PROGRESS.md               # Nhật ký theo dõi tiến độ (16/16 task), kết quả chính & quyết định kỹ thuật
```

### 1.2. Luồng xử lý dữ liệu (Data Pipeline)

```mermaid
flowchart TD
    A["1. Nạp CSV & Kiểm tra toàn vẹn<br/>(13.611 mẫu, 17 cột)"] --> B["2. Tiền xử lý<br/>(0 thiếu, bỏ 68 trùng -> 13.543 mẫu,<br/>báo cáo IQR giữ nguyên ngoại lai)"]
    B --> C["3. EDA & Thống kê mô tả<br/>(Biểu đồ 01 - 05, phát hiện 21 cặp |r| > 0.9)"]
    C --> D["4. Phân loại có giám sát (RQ1)<br/>(Dữ liệu thô, chia 80/20 stratify)"]
    C --> E["5. Phân cụm không giám sát (RQ2)<br/>(StandardScaler, bỏ cột Class)"]
    D --> D1["GaussianNB & DecisionTreeClassifier<br/>trên 3 bộ đặc trưng (6 tổ hợp)"]
    D1 --> D2["5-fold Stratified CV (Train) chọn mô hình tốt nhất<br/>-> Đánh giá Test set 20% (Biểu đồ 06 - 08)"]
    E --> E1["K-Means (K = 2..10)<br/>Elbow & Silhouette (Biểu đồ 09)"]
    E1 --> E2["K-Means K = 7, Crosstab, ARI<br/>& Chiếu PCA 2D (Biểu đồ 10 - 12)"]
    D2 --> F["6. Tổng kết RQ1, RQ2 & Nghiệm thu"]
    E2 --> F
```

### 1.3. Môi trường & Công nghệ

| Thành phần | Công nghệ / Thư viện | Vai trò trong dự án |
|---|---|---|
| Ngôn ngữ lập trình | **Python 3.10+** | Ngôn ngữ triển khai chính toàn bộ pipeline |
| Thư viện chuẩn Python | `os`, `sys`, `warnings` | Kiểm tra file, tạo thư mục `outputs/`, dừng an toàn `sys.exit(1)` khi lỗi |
| Xử lý dữ liệu & Toán học | **Pandas**, **NumPy** | Đọc CSV, lọc trùng lặp, tính phân vị IQR, thống kê mô tả, ma trận tương quan |
| Học máy (Machine Learning) | **Scikit-Learn** | `GaussianNB`, `DecisionTreeClassifier`, `KMeans`, `StandardScaler`, `PCA`, `train_test_split`, `StratifiedKFold`, `cross_validate`, `classification_report`, `confusion_matrix`, `silhouette_score`, `adjusted_rand_score` |
| Trực quan hóa | **Matplotlib**, **Seaborn** | Style `seaborn-v0_8-whitegrid`, font `DejaVu Sans`, bảng màu `tab10`, xuất ảnh PNG 300 DPI |

---

## 2. Hai câu hỏi nghiên cứu cốt lõi (Research Questions)

| Mã | Loại bài toán | Câu hỏi nghiên cứu | Phương pháp giải quyết |
|---|---|---|---|
| **RQ1** | **Phân loại** (*Supervised*) | Dựa vào nhóm đặc trưng nào — **kích thước** (`Area`, `Perimeter`, ...) hay **hình dạng** (`roundness`, `ShapeFactor`, ...) — để xác định loại đậu chính xác hơn? | • Phân chia Train/Test tỷ lệ `80/20` có `stratify=y` và `RANDOM_STATE = 42` (giữ nguyên dữ liệu thô, không chuẩn hóa).<br>• Phân tách 16 đặc trưng thành **Nhóm Kích thước (8 đặc trưng)** và **Nhóm Hình dạng (8 đặc trưng)**.<br>• Huấn luyện và đánh giá **6 tổ hợp** (2 thuật toán `GaussianNB`, `DecisionTreeClassifier` $\times$ 3 bộ đặc trưng: *Toàn bộ 16*, *Kích thước 8*, *Hình dạng 8*) bằng **5-fold Stratified CV** trên tập Train.<br>• Chọn mô hình tốt nhất theo **F1-macro CV trung bình** (tránh rò rỉ dữ liệu từ tập Test), đánh giá một lần cuối trên tập Test 20% và phân tích `feature_importances_` (Gini Importance). |
| **RQ2** | **Phân cụm** (*Unsupervised*) | Số cụm tối ưu của **K-Means** là bao nhiêu, và các cụm tìm được có khớp với **7 loại đậu thật** không? | • Chuẩn hóa toàn bộ 16 đặc trưng số trên $13.543$ mẫu sạch bằng `StandardScaler` (tuyệt đối tách cột `Class` khỏi mô hình).<br>• Chạy `KMeans` (`n_init=10`, `random_state=42`) với $K \in [2..10]$.<br>• Đánh giá và chọn $K$ tối ưu dựa trên đường cong **Elbow (Inertia/WCSS)** và **Silhouette Score**.<br>• Đối chiếu cấu trúc cụm tại $K = 7$ với 7 nhãn thực tế bằng ma trận chéo **Crosstab**, chỉ số **Adjusted Rand Index (ARI)** và giảm chiều **PCA 2D** để trực quan hóa phân bố cụm. |

---

## 3. Tóm tắt kết quả thực nghiệm chính

> Toàn bộ số liệu dưới đây được ghi nhận trực tiếp từ kết quả chạy thực nghiệm của [`main.py`](main.py) và tổng hợp trong [`Planing/PROGRESS.md`](Planing/PROGRESS.md) (định dạng 4 chữ số thập phân `.4f`).

### 3.1. Thống kê dữ liệu, Tiền xử lý & EDA
- **Dữ liệu gốc:** `13.611` mẫu $\times$ `17` cột (`16` đặc trưng số: 2 `int64`, 14 `float64`; `1` cột nhãn `Class` với 7 lớp: `DERMASON` 3.546, `SIRA` 2.636, `SEKER` 2.027, `HOROZ` 1.928, `CALI` 1.630, `BARBUNYA` 1.322, `BOMBAY` 522).
- **Làm sạch dữ liệu:**
  - Giá trị khuyết thiếu (NaN): **0** giá trị thiếu.
  - Dữ liệu trùng lặp: Phát hiện và loại bỏ **68 dòng trùng lặp hoàn toàn** (thuộc lớp `HOROZ`, giảm còn 1.860 mẫu), còn lại **13.543 mẫu sạch** (`Train = 10.834` mẫu [80,00%], `Test = 2.709` mẫu [20,00%]).
- **Phát hiện ngoại lai bằng IQR ($1.5 \times \text{IQR}$):**
  - Tổng cộng **3.004 / 13.543 mẫu (22,18%)** có ít nhất 1 đặc trưng vượt ngưỡng IQR (cao nhất ở `Eccentricity`: 833 mẫu [6,15%], `Solidity`: 774 mẫu [5,72%], `ShapeFactor4`: 760 mẫu [5,61%], `Area`: 551 mẫu [4,07%]; thấp nhất ở `ShapeFactor2`: 0 mẫu [0,00%]).
  - **Quyết định kỹ thuật:** Giữ nguyên toàn bộ 3.004 mẫu ngoại lai vì đây là biến thiên sinh học tự nhiên giữa các giống đậu (đặc biệt giống hạt lớn `BOMBAY`), không phải nhiễu đo lường.
- **Đa cộng tuyến cao ($|r| > 0.9$):** Phát hiện **21 cặp đặc trưng** có tương quan Pearson rất mạnh, nổi bật nhất là `Area` – `ConvexArea` ($r = 0.9999$), `Compactness` – `ShapeFactor3` ($r = 0.9987$), `Perimeter` – `EquivDiameter` ($r = 0.9915$) và `AspectRation` – `Compactness` ($r = -0.9876$).

### 3.2. Kết quả RQ1: Phân loại 7 giống đậu khô

#### Bảng so sánh 6 tổ hợp mô hình (5-fold Stratified CV & Tập kiểm tra 20%)

| Thuật toán | Bộ đặc trưng | F1-macro CV (5-fold) | Accuracy CV (5-fold) | Test Accuracy | Test F1-macro |
|---|---|---:|---:|---:|---:|
| `GaussianNB` | Toàn bộ (16 đặc trưng) | `0.7632 ± 0.0085` | `0.7620 ± 0.0073` | `0.7630` | `0.7669` |
| `GaussianNB` | Kích thước (8 đặc trưng) | `0.7625 ± 0.0080` | `0.7612 ± 0.0068` | `0.7619` | `0.7659` |
| `GaussianNB` | Hình dạng (8 đặc trưng) | `0.9062 ± 0.0067` | `0.8916 ± 0.0082` | `0.8915` | `0.9060` |
| `Decision Tree` | Toàn bộ (16 đặc trưng) | `0.9107 ± 0.0046` | `0.8960 ± 0.0058` | `0.8955` | `0.9098` |
| `Decision Tree` | Kích thước (8 đặc trưng) | `0.8878 ± 0.0056` | `0.8756 ± 0.0055` | `0.8808` | `0.8920` |
| **`Decision Tree`** | **Hình dạng (8 đặc trưng)** | **`0.9128 ± 0.0072`** | **`0.8984 ± 0.0072`** | **`0.8963`** | **`0.9099`** |

#### Báo cáo phân loại chi tiết trên Tập kiểm tra (Test Set = 2.709 mẫu) của mô hình tốt nhất (`Decision Tree - Hình dạng`)

| Lớp (`Class`) | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `BARBUNYA` | `0.9023` | `0.8717` | `0.8868` | 265 |
| `BOMBAY` | `1.0000` | `1.0000` | `1.0000` | 104 |
| `CALI` | `0.8820` | `0.9172` | `0.8992` | 326 |
| `DERMASON` | `0.9046` | `0.8956` | `0.9001` | 709 |
| `HOROZ` | `0.9270` | `0.9220` | `0.9245` | 372 |
| `SEKER` | `0.9293` | `0.9384` | `0.9338` | 406 |
| `SIRA` | `0.8239` | `0.8254` | `0.8246` | 527 |
| **Accuracy / Macro Avg** | **`0.9099`** | **`0.9101`** | **`0.8963` / `0.9099`** | **2.709** |

- **Phân tích lỗi nhầm lẫn & Trọng số đặc trưng:**
  - `BOMBAY` đạt độ chính xác tuyệt đối `1.0000` ($104/104$ mẫu); cặp nhầm lẫn lớn nhất là `DERMASON` $\leftrightarrow$ `SIRA` với tổng **116 mẫu nhầm lẫn 2 chiều** (`DERMASON` $\rightarrow$ `SIRA`: 65 mẫu; `SIRA` $\rightarrow$ `DERMASON`: 51 mẫu).
  - Trên mô hình Decision Tree toàn bộ 16 đặc trưng, **Nhóm Hình dạng chiếm `0.5995` (59,95%)** tổng Gini Importance (gấp **1,50 lần** nhóm Kích thước với `0.4005` / 40,05%); hai đặc trưng dẫn đầu đều thuộc nhóm Hình dạng là `ShapeFactor1` (`0.2293`) và `Compactness` (`0.1883`).
- **Kết luận RQ1:** **Nhóm đặc trưng Hình dạng** giúp xác định loại đậu chính xác hơn rõ rệt so với nhóm Kích thước (tăng **+14,37%** F1-macro CV trên `GaussianNB` và **+2,50%** F1-macro CV trên `Decision Tree`), nhờ các tỷ lệ hình thái không thứ nguyên phản ánh ổn định đặc tính di truyền của giống loài.

### 3.3. Kết quả RQ2: Phân cụm không giám sát K-Means & Đối chiếu PCA 2D

#### Kết quả K-Means theo số cụm $K \in [2..10]$ (trên 13.543 mẫu chuẩn hóa `StandardScaler`)

| Số cụm ($K$) | Inertia (WCSS) | Silhouette Score | Nhận xét |
|---:|---:|---:|---|
| 2 | `129194.0199` | `0.3972` | Phân tách 2 khối kích thước lớn/nhỏ |
| **3** | **`94331.7346`** | **`0.4044`** | **Số cụm $K$ tối ưu (Silhouette cao nhất, điểm gãy Elbow)** |
| 4 | `75993.1949` | `0.3423` | Tốc độ giảm Inertia bắt đầu thoải dần |
| 5 | `61624.6374` | `0.3572` | Độ dốc Elbow phẳng dần từ $K = 3..5$ |
| 6 | `54900.7483` | `0.3599` | |
| **7** | **`48671.4570`** | **`0.3088`** | **Mốc đối chiếu tương ứng 7 giống đậu thật ($\text{ARI} = 0.6669$)** |
| 8 | `45083.3590` | `0.3027` | |
| 9 | `41904.8161` | `0.3029` | |
| 10 | `39707.1259` | `0.2681` | Phân mảnh cụm, Silhouette giảm mạnh |

- **Độ khớp với 7 lớp thật tại $K = 7$:**
  - Chỉ số tương đồng phân cụm so với nhãn thật: **$\text{ARI} = 0.6669$** (mức khớp khá cao giữa cấu trúc hình thái tự nhiên và phân loại sinh học).
  - Độ tinh khiết nổi bật trên từng cụm: `Cụm 3` $\rightarrow$ `BOMBAY` (**100,00%** thuần nhất, bao phủ 99,62% lớp); `Cụm 2` $\rightarrow$ `HOROZ` (**93,83%**); `Cụm 0` $\rightarrow$ `DERMASON` (**92,21%**); `Cụm 5` $\rightarrow$ `SEKER` (**92,18%**); `Cụm 6` $\rightarrow$ `SIRA` (**74,71%**, có 499 mẫu `DERMASON` giao thoa); `Cụm 4` gom chung hai giống hạt lớn bầu dục `CALI` (1.302 mẫu) và `BARBUNYA` (1.141 mẫu).
- **Không gian giảm chiều PCA 2D:** `PC1` giải thích **55,55%** phương sai và `PC2` giải thích **26,36%** phương sai (tổng cộng bảo toàn **81,91%** thông tin từ 16 đặc trưng gốc).
- **Kết luận RQ2:** Số cụm tối ưu tự nhiên của K-Means là **$K = 3$** ($\text{Silhouette} = 0.4044$), hội tụ về 3 đại nhóm hình thái (hạt khổng lồ, hạt thuôn dài/lớn, hạt nhỏ/tròn). Khi phân thành $K = 7$, các cụm khớp khá tốt với 7 giống đậu thực tế ($\text{ARI} = 0.6669$), tái tạo thuần khiết $4/7$ giống đậu riêng biệt.

---

## 4. Danh mục 12 biểu đồ trực quan hóa trong `outputs/`

Toàn bộ biểu đồ được xuất tự động ở định dạng PNG, độ phân giải **300 DPI**, `bbox_inches="tight"`, sử dụng style `seaborn-v0_8-whitegrid`, font `DejaVu Sans` và bảng màu `tab10` cố định theo lớp quy định trong [`Planing/DESIGN.md`](Planing/DESIGN.md).

| STT | Tên file trong `outputs/` | Nhóm | Nội dung & Ý nghĩa phân tích |
|---:|---|---|---|
| 01 | [`outputs/01_phan_bo_lop.png`](outputs/01_phan_bo_lop.png) | Tổng quan EDA | Biểu đồ cột phân bố số lượng mẫu của 7 lớp đậu sạch (kèm số đếm trên đỉnh cột), phản ánh sự mất cân bằng nhẹ giữa lớp lớn nhất (`DERMASON`: 3.546) và nhỏ nhất (`BOMBAY`: 522). |
| 02 | [`outputs/02_histogram_dac_trung.png`](outputs/02_histogram_dac_trung.png) | Đơn biến EDA | Lưới $4 \times 4$ Histogram kèm đường cong mật độ KDE cho 16 đặc trưng số, nhận diện phân bố lệch phải của nhóm kích thước và phân bố đa đỉnh của các hệ số hình dạng. |
| 03 | [`outputs/03_boxplot_ngoai_lai.png`](outputs/03_boxplot_ngoai_lai.png) | Đơn biến EDA | Lưới $4 \times 4$ Boxplot cho 16 đặc trưng số, trực quan hóa 3.004 mẫu ngoại lai IQR (22,18%) nhằm chứng minh tính hợp lý sinh học của quyết định giữ nguyên ngoại lai. |
| 04 | [`outputs/04_ma_tran_tuong_quan.png`](outputs/04_ma_tran_tuong_quan.png) | Đa biến EDA | Heatmap ma trận tương quan Pearson (bảng màu `coolwarm`, giá trị 2 chữ số thập phân), phơi bày 21 cặp đa cộng tuyến cao ($|r| > 0.9$). |
| 05 | [`outputs/05_boxplot_theo_lop.png`](outputs/05_boxplot_theo_lop.png) | Đa biến EDA | So sánh phân bố 3 đặc trưng tiêu biểu (`Area`, `AspectRation`, `ShapeFactor1`) qua 7 lớp đậu, minh họa khả năng tách lớp riêng biệt của từng nhóm đặc trưng. |
| 06 | [`outputs/06_confusion_matrix.png`](outputs/06_confusion_matrix.png) | Phân loại (RQ1) | Heatmap ma trận nhầm lẫn (bảng màu `Blues`) của mô hình tốt nhất (`Decision Tree - Hình dạng`) trên tập Test 2.709 mẫu, chỉ rõ điểm mạnh tuyệt đối trên `BOMBAY` và vùng nhầm lẫn `DERMASON` – `SIRA`. |
| 07 | [`outputs/07_so_sanh_mo_hinh.png`](outputs/07_so_sanh_mo_hinh.png) | Phân loại (RQ1) | Biểu đồ cột nhóm so sánh trực diện F1-macro (5-fold CV) giữa `GaussianNB` và `Decision Tree` trên Nhóm Kích thước và Nhóm Hình dạng, làm nổi bật bước nhảy hiệu năng +14,37% trên `GaussianNB`. |
| 08 | [`outputs/08_feature_importances.png`](outputs/08_feature_importances.png) | Phân loại (RQ1) | Biểu đồ cột ngang xếp hạng `feature_importances_` (Gini Importance) của mô hình Decision Tree tốt nhất (Nhóm Hình dạng), xác định `ShapeFactor1` (`0.3798`) và `ShapeFactor3` (`0.2551`) là hai nhân tố quyết định hàng đầu. |
| 09 | [`outputs/09_elbow_silhouette.png`](outputs/09_elbow_silhouette.png) | Phân cụm (RQ2) | Biểu đồ kép đường cong Elbow (Inertia) và Silhouette Score theo $K \in [2..10]$, đánh dấu $K = 3$ tối ưu (`0.4044`) và đối chiếu mốc thực tế $K = 7$ (`0.3088`). |
| 10 | [`outputs/10_crosstab_cum_lop.png`](outputs/10_crosstab_cum_lop.png) | Phân cụm (RQ2) | Heatmap ma trận chéo (Crosstab) đối chiếu 7 cụm K-Means với 7 lớp đậu thật (bảng màu `YlGnBu`), trực quan hóa độ tinh khiết cao của `BOMBAY`, `HOROZ`, `DERMASON`, `SEKER` và sự gộp cụm `CALI` – `BARBUNYA`. |
| 11 | [`outputs/11_pca_cum.png`](outputs/11_pca_cum.png) | Phân cụm (RQ2) | Biểu đồ phân tán 7 cụm K-Means và tâm cụm (`X`) chiếu lên không gian PCA 2D (giải thích 81,91% phương sai dữ liệu). |
| 12 | [`outputs/12_pca_lop_that.png`](outputs/12_pca_lop_that.png) | Phân cụm (RQ2) | Biểu đồ phân tán 7 lớp đậu thật trên cùng hệ trục tọa độ PCA 2D, đặt đối chiếu trực tiếp với `11_pca_cum.png` để kiểm chứng mức độ tương đồng không gian phân cụm ($\text{ARI} = 0.6669$). |

*(Lưu ý: Khi chạy `main.py`, thư mục `outputs/` còn sinh các tệp tương thích ngược `02_histogram_kde.png`, `03_boxplot.png`, `04_heatmap_tuong_quan.png` đồng nhất nội dung với các biểu đồ 02, 03, 04 tương ứng).*

---

## 5. Hướng dẫn cài đặt & Tái lập thực nghiệm (Reproducibility)

### 5.1. Yêu cầu môi trường & Cài đặt thư viện
Đảm bảo máy đã cài đặt **Python 3.10 trở lên**. Khuyến nghị cài đặt các thư viện cần thiết bằng `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 5.2. Chuẩn bị dữ liệu đầu vào
Đặt tệp dữ liệu gốc **`dry_beans.csv`** vào ngay thư mục gốc của dự án (cùng cấp với [`main.py`](main.py)). Nếu thiếu tệp hoặc thiếu bất kỳ cột nào trong 17 cột bắt buộc, chương trình sẽ tự động dừng an toàn với mã thoát `1` kèm thông báo hướng dẫn khắc phục bằng tiếng Việt.

### 5.3. Thực thi chương trình
Chạy duy nhất một lệnh từ thư mục gốc `Data-Mining-09/`:

```bash
python main.py
```

- **Chế độ tương tác mặc định:** Chương trình sẽ hiển thị lần lượt các cửa sổ biểu đồ EDA (`01` – `05`) qua `plt.show()`. Đóng từng cửa sổ để pipeline tiếp tục chạy các bước tiếp theo và xuất trọn bộ 12 biểu đồ PNG 300 DPI vào `outputs/`.
- **Chế độ chạy tự động / không mở cửa sổ đồ họa (Headless):**
  - Trên Windows PowerShell:
    ```powershell
    $env:MPLBACKEND="Agg"; python main.py
    ```
  - Trên Linux/macOS:
    ```bash
    MPLBACKEND=Agg python main.py
    ```

### 5.4. Cam kết tính tất định & Tái lập 100%
Toàn bộ các thành phần có yếu tố ngẫu nhiên trong pipeline đều được cố định với hằng số **`RANDOM_STATE = 42`**:
- Phân chia tập Train/Test `80/20` (`train_test_split(..., stratify=y, random_state=42)`).
- Kiểm định chéo 5-fold (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).
- Huấn luyện cây quyết định (`DecisionTreeClassifier(random_state=42)`).
- Phân cụm K-Means (`KMeans(..., random_state=42, n_init=10)`).
- Giảm chiều trực quan hóa (`PCA(n_components=2, random_state=42)`).

Kết quả kiểm chứng ở Task T14 xác nhận khi chạy `python main.py` hai lần liên tiếp độc lập, toàn bộ **961 dòng output console** và mọi chỉ số đánh giá trùng khớp **100%** không sai lệch.

---

## 6. Tác giả & Giấy phép

- **Nhóm thực hiện:** Nhóm 09 — Dự án [`Data-Mining-09`](README.md) *(Thành viên: [Điền Họ tên & MSSV các thành viên])*
- **Học phần:** Khai phá Dữ liệu (*Data Mining*)
- **Đơn vị đào tạo:** *[Điền tên Khoa / Trường Đại học]*
- **Nguồn dữ liệu:** *Dry Bean Dataset* (Koklu, M. & Ozkan, I. A., 2020 — *Multiclass classification of dry beans using computer vision and machine learning techniques*, Computers and Electronics in Agriculture, 174, 105507).
- **Giấy phép:** Dự án được xây dựng phục vụ mục đích học tập, nghiên cứu học thuật và báo cáo đồ án môn học.
