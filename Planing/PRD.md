# PRD - Khai phá dữ liệu: Phân loại đậu khô (Dry Bean Dataset)

## 1. Giới thiệu và mục tiêu
- **Đề tài:** Khai phá dữ liệu, phân loại đậu khô (Dry Bean Dataset).
- **Mục tiêu:**
  - Áp dụng quy trình khai phá dữ liệu hoàn chỉnh: tiền xử lý, EDA, mô hình hóa, đánh giá.
  - Xây dựng mô hình phân loại 7 loại đậu từ 16 đặc trưng hình thái.
  - Khám phá cấu trúc tự nhiên của dữ liệu bằng phân cụm.
- **Phạm vi:** một file `main.py` chạy trọn luồng. Không triển khai ứng dụng hay giao diện.

## 2. Bài toán giải quyết
- **Dữ liệu:** 13.611 mẫu, 16 đặc trưng số (kích thước và hình dạng), 1 nhãn `Class` gồm 7 lớp: Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira.
- **Phân loại:** dự đoán loại đậu từ đặc trưng hình thái.
- **Phân cụm:** tìm nhóm tự nhiên trong dữ liệu mà không dùng nhãn.
- **Thỏa tiêu chí guideline:** ≥ 200 dòng, ≥ 5 cột, có nhãn đầu ra để kết luận.

## 3. Đối tượng hướng tới
Giảng viên và hội đồng chấm đồ án. Nội dung viết theo hướng học thuật, bám sát guideline, giải thích rõ phương pháp và kết quả.

## 4. Câu hỏi nghiên cứu

| Mã | Loại | Câu hỏi |
|---|---|---|
| **RQ1** | Phân loại | Dựa vào nhóm đặc trưng nào, kích thước (Area, Perimeter, ...) hay hình dạng (Roundness, ShapeFactor, ...), để xác định loại đậu chính xác hơn? |
| **RQ2** | Phân cụm | Số cụm tối ưu của K-Means là bao nhiêu, và các cụm có khớp với 7 loại đậu thật không? |

- **RQ1:**
  - Dùng 3 phép thử: toàn bộ đặc trưng, chỉ nhóm kích thước, chỉ nhóm hình dạng.
  - Mô hình: Naive Bayes (GaussianNB) và Decision Tree (đối chứng).
  - Chỉ số: Accuracy và F1-macro trên tập kiểm tra và 5-fold CV, kèm `feature_importances_` của Decision Tree.
- **RQ2:**
  - K từ 2 đến 10, đánh giá bằng Elbow và Silhouette.
  - Độ khớp với nhãn thật: crosstab và ARI.

## 5. Quyết định phương pháp đã chốt

| Hạng mục | Quyết định |
|---|---|
| Thuật toán | Naive Bayes, Decision Tree, K-Means (3 thuật toán, đúng khoảng 2-3 của guideline) |
| Ngoại lai | Phát hiện bằng IQR, giữ nguyên, giải thích lý do |
| Dữ liệu trùng và thiếu | Loại bỏ trùng, kiểm tra thiếu theo guideline |
| Chuẩn hóa | `StandardScaler` chỉ cho K-Means |
| Chia dữ liệu | 80/20 có `stratify`, thêm 5-fold stratified CV |
| Tái lập | `RANDOM_STATE = 42` |
| Bảng màu / xuất biểu đồ | `tab10`, PNG DPI 300 vào `outputs/`, có `plt.show()` |
| Xử lý lỗi | Dừng chương trình, thông báo tiếng Việt kèm hướng khắc phục |

## 6. Tiêu chí nghiệm thu
- `main.py` chạy từ đầu đến cuối không lỗi trên `dry_beans.csv`.
- Có đủ thống kê mô tả, biểu đồ đơn biến, đa biến và heatmap tương quan.
- Trả lời được RQ1 và RQ2 bằng số liệu và biểu đồ.
- Có bước "chọn thuật toán tốt nhất": so sánh Naive Bayes và Decision Tree cho RQ1.
- Kết quả giống nhau giữa các lần chạy.

## 7. Ngoài phạm vi
Tinh chỉnh siêu tham số sâu, thuật toán ngoài danh sách đã chốt, triển khai thực tế, giao diện web.
