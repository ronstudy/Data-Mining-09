# TASKS - Danh sách công việc

4 sprint, 16 task. Mỗi task kiểm tra được bằng một kết quả cụ thể.

## Sprint 1: Chuẩn bị, nạp và tiền xử lý
| ID | Task | Kiểm tra |
|---|---|---|
| T1 | Đặt `dry_beans.csv` vào thư mục gốc, cài thư viện, tạo khung `main.py` (import, hằng số, `main()`) | `main.py` chạy không lỗi |
| T2 | `load_data`: đọc CSV, kiểm tra file và cột (báo lỗi tiếng Việt), in kích thước, kiểu dữ liệu, số lớp | Khớp 13.611 mẫu, 17 cột, 7 lớp. Thiếu file thì báo đúng thông báo |
| T3 | `preprocess`: kiểm tra thiếu, loại dòng trùng, phát hiện ngoại lai bằng IQR (giữ nguyên), tạo hai nhóm đặc trưng kích thước và hình dạng | Có báo cáo thiếu, trùng, ngoại lai. Đủ 16 đặc trưng ở hai nhóm |

## Sprint 2: EDA và thống kê
| ID | Task | Kiểm tra |
|---|---|---|
| T4 | Thống kê mô tả (mean, median, std, phân vị) và biểu đồ 01 phân bố lớp | Bảng console, 1 PNG |
| T5 | Biểu đồ đơn biến: 02 histogram + KDE, 03 boxplot | 2 PNG lưới 4×4 |
| T6 | Biểu đồ đa biến: 04 heatmap tương quan (in cặp \|r\| > 0,9), 05 boxplot theo lớp. In nhận xét EDA | 2 PNG, danh sách cặp tương quan cao |

## Sprint 3: Mô hình hóa
| ID | Task | Kiểm tra |
|---|---|---|
| T7 | Chia 80/20 có `stratify`. Huấn luyện GaussianNB và Decision Tree trên 3 bộ đặc trưng | Tỉ lệ lớp giữ nguyên, có kết quả 6 tổ hợp |
| T8 | 5-fold stratified CV cho 6 tổ hợp, chọn mô hình tốt nhất theo F1-macro CV | Bảng trung bình ± độ lệch chuẩn, nêu rõ mô hình được chọn |
| T9 | Đánh giá trên tập kiểm tra, vẽ 06 Confusion Matrix, 07 so sánh mô hình, 08 `feature_importances_` | 3 PNG, classification report. Chỉ ra cặp lớp hay nhầm |
| T10 | Kết luận RQ1: nhóm kích thước hay hình dạng quan trọng hơn | Câu trả lời dựa trên số liệu |
| T11 | `StandardScaler` (bỏ `Class`), KMeans K từ 2 đến 10, vẽ 09 Elbow + Silhouette, chọn K tối ưu kèm lý do | Bảng 9 dòng, 1 PNG |
| T12 | Fit KMeans với K tối ưu, tính crosstab và ARI, vẽ 10, 11 (PCA 2D) và 12 (heatmap crosstab) | 3 PNG, bảng crosstab, chỉ số ARI |
| T13 | Kết luận RQ2: K tối ưu và độ khớp với 7 lớp | Câu trả lời dựa trên số liệu |

## Sprint 4: Nghiệm thu
| ID | Task | Kiểm tra |
|---|---|---|
| T14 | Chạy `main.py` hai lần, so sánh kết quả | Chạy không lỗi, số liệu giống nhau |
| T15 | Rà PEP8 và `RULES.md`, kiểm tra đủ 12 biểu đồ và DPI 300 | 12 file trong `outputs/`, không vi phạm điều cấm |
| T16 | Đối chiếu tiêu chí nghiệm thu trong `PRD.md` và chuẩn bị số liệu cho báo cáo 7 phần | Tick đủ các mục |
