# DESIGN - Tiêu chuẩn trực quan hóa

## 1. Cấu hình chung
- **Style:** `seaborn-v0_8-whitegrid`. Font `DejaVu Sans` (có sẵn, hỗ trợ tiếng Việt có dấu).
- **Cỡ chữ:** tiêu đề 14 (đậm), nhãn trục 11, tick 10, chú thích 10.
- **Kích thước:** biểu đồ đơn 8×6 inch, lưới nhiều ô 16×12 inch, heatmap 12×10 inch.
- **Xuất file:** PNG, `dpi=300`, `bbox_inches="tight"`, lưu vào `outputs/`. Sau đó gọi `plt.show()`.
- **Ngôn ngữ nhãn:** tiêu đề và nhãn trục bằng tiếng Việt, tên lớp giữ nguyên tiếng Anh (BARBUNYA...).
- **Đặt tên file:** `NN_ten_bieu_do.png`, ví dụ `01_phan_bo_lop.png`.

## 2. Bảng màu
- **Màu theo lớp:** `tab10`, gán cố định theo thứ tự tên lớp.

| Lớp | Chỉ số `tab10` |
|---|---|
| BARBUNYA | 0 |
| BOMBAY | 1 |
| CALI | 2 |
| DERMASON | 3 |
| HOROZ | 4 |
| SEKER | 5 |
| SIRA | 6 |

- **Heatmap tương quan:** `coolwarm`, `vmin=-1`, `vmax=1`, `center=0`.
- **Confusion Matrix:** `Blues`.
- **Cụm K-Means:** cũng dùng `tab10` nhưng vẽ bằng marker khác nhau (hình tròn cho cụm), để không nhầm với màu của lớp thật.

## 3. Danh mục biểu đồ

| STT | Nhóm | Biểu đồ | Ghi chú |
|---|---|---|---|
| 01 | Tổng quan | Phân bố lớp (bar) | Có số lượng trên mỗi cột |
| 02 | Đơn biến | Histogram + KDE cho 16 đặc trưng | Lưới 4×4 |
| 03 | Đơn biến | Boxplot cho 16 đặc trưng | Lưới 4×4, thấy rõ ngoại lai |
| 04 | Đa biến | Heatmap tương quan | Ghi giá trị `annot`, 2 chữ số |
| 05 | Đa biến | Boxplot một đặc trưng theo lớp | Chọn 2-3 đặc trưng tiêu biểu |
| 06 | RQ1 | Confusion Matrix (mô hình tốt nhất) | Chuẩn hóa theo hàng, thêm số đếm |
| 07 | RQ1 | So sánh Accuracy / F1 theo mô hình và bộ đặc trưng | Bar nhóm |
| 08 | RQ1 | `feature_importances_` của Decision Tree | Bar ngang, sắp xếp giảm dần, tô màu theo nhóm kích thước/hình dạng |
| 09 | RQ2 | Elbow và Silhouette theo K | Hai biểu đồ cạnh nhau, đánh dấu K tối ưu |
| 10 | RQ2 | Cụm K-Means trên PCA 2D | Có tâm cụm và tỷ lệ phương sai giải thích |
| 11 | RQ2 | PCA 2D tô theo nhãn thật | Đặt cạnh biểu đồ 10 để so sánh |
| 12 | RQ2 | Heatmap crosstab cụm và lớp | Dùng `Blues`, có số đếm |

## 4. Quy tắc trình bày
- Mọi biểu đồ có tiêu đề, nhãn trục (kèm đơn vị nếu có), chú thích nếu có nhiều màu.
- Không dùng 3D, không dùng biểu đồ tròn.
- Tên đặc trưng giữ nguyên tiếng Anh như trong CSV (ví dụ `Area`, `Roundness`).
- Với lưới 16 đặc trưng, dùng `tight_layout()` để tránh chồng chữ.
- Điểm dữ liệu trên scatter: `alpha=0.5`, `s=10` để thấy mật độ.
- `plt.show()` chặn chương trình đến khi đóng cửa sổ biểu đồ. Cần đóng từng cửa sổ để chương trình chạy tiếp.
