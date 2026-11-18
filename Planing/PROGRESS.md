# PROGRESS - Theo dõi tiến độ

## 1. Tổng quan
| Hạng mục | Giá trị |
|---|---|
| Giai đoạn hiện tại | Đang thực hiện Sprint 3 (Mô hình hóa) |
| Tiến độ | 10/16 task |
| Cập nhật lần cuối | 2026-11-19 |

## 2. Tình trạng tài liệu
| File | Trạng thái |
|---|---|
| PRD.md | Hoàn tất |
| ARCHITECTURE.md | Hoàn tất |
| DESIGN.md | Hoàn tất |
| RULES.md | Hoàn tất |
| TASKS.md | Hoàn tất |
| PROGRESS.md | Hoàn tất |

## 3. Bảng theo dõi task
Ký hiệu: ⬜ Chưa làm, 🔄 Đang làm, ✅ Xong, ❌ Bị chặn.

| Sprint | ID | Task (tóm tắt) | Trạng thái | Ngày xong | Ghi chú |
|---|---|---|---|---|---|
| 1 | T1 | Khung dự án, dữ liệu, thư viện | ✅ | 2026-11-17 | Khung main.py duy nhất, hằng số, điểm vào main, thư viện hợp lệ |
| 1 | T2 | `load_data` | ✅ | 2026-11-17 | Nạp 13.611 mẫu, 17 cột, 7 lớp; kiểm tra file và cột, báo lỗi tiếng Việt |
| 1 | T3 | `preprocess` | ✅ | 2026-11-17 | Bỏ 68 dòng trùng (còn 13.543 mẫu), 0 thiếu, 3.004 mẫu ngoại lai IQR (giữ nguyên), tách 2 nhóm đặc trưng |
| 2 | T4 | Thống kê mô tả, biểu đồ 01 | ✅ | 2026-11-17 | Bảng thống kê mô tả 16 đặc trưng và xuất ảnh 01_phan_bo_lop.png (DPI 300) |
| 2 | T5 | Biểu đồ đơn biến 02, 03 | ✅ | 2026-11-17 | Xuất lưới 4x4 biểu đồ 02 (Histogram+KDE) và 03 (Boxplot) cho 16 đặc trưng (DPI 300) |
| 2 | T6 | Biểu đồ đa biến 04, 05 | ✅ | 2026-11-17 | Heatmap tương quan (21 cặp \|r\| > 0.9), boxplot 3 đặc trưng theo lớp, nhận xét EDA |
| 3 | T7 | Chia dữ liệu, huấn luyện 6 tổ hợp | ✅ | 2026-11-17 | Chia 80/20 stratify (10.834 Train, 2.709 Test), huấn luyện 6 tổ hợp mô hình |
| 3 | T8 | 5-fold CV, chọn mô hình | ✅ | 2026-11-17 | 5-fold CV 6 tổ hợp; chọn Decision Tree - Hình dạng (F1 = 0.9128 ± 0.0072) |
| 3 | T9 | Đánh giá, biểu đồ 06-08 | ✅ | 2026-11-17 | Đánh giá Test set (Acc=0.8963, F1=0.9099), xuất biểu đồ 06, 07, 08 |
| 3 | T10 | Kết luận RQ1 | ✅ | 2026-11-19 | Nhóm Hình dạng vượt trội (59,95% importance; F1 CV DT: 0.9128 vs 0.8878; GNB: 0.9062 vs 0.7625) |
| 3 | T11 | K-Means, Elbow, Silhouette | ⬜ | | |
| 3 | T12 | Crosstab, ARI, biểu đồ 10-12 | ⬜ | | |
| 3 | T13 | Kết luận RQ2 | ⬜ | | |
| 4 | T14 | Chạy hai lần, kiểm tra tái lập | ⬜ | | |
| 4 | T15 | Rà PEP8, RULES, biểu đồ | ⬜ | | |
| 4 | T16 | Đối chiếu PRD, chuẩn bị báo cáo | ⬜ | | |

## 4. Bug Tracker
| ID | Ngày | Task liên quan | Mô tả lỗi | Mức độ | Trạng thái | Cách xử lý |
|---|---|---|---|---|---|---|
| (trống) | | | | | | |

## 5. Kết quả chính (điền khi có số liệu thực)
| Chỉ số | Giá trị |
|---|---|
| Số mẫu sau khi loại trùng | 13.543 |
| Mô hình tốt nhất (F1-macro CV) | Decision Tree - Hình dạng (8 đặc trưng) (F1 = 0.9128 ± 0.0072) |
| Accuracy / F1-macro trên tập kiểm tra | Accuracy = 0.8963 / F1-macro = 0.9099 (Decision Tree - Hình dạng) |
| Nhóm đặc trưng quan trọng hơn | Nhóm Hình dạng (59,95% importance; F1 CV DT: 0.9128 vs 0.8878; GNB: 0.9062 vs 0.7625) |
| F1-macro CV theo bộ đặc trưng (Decision Tree) | Hình dạng: 0.9128 ± 0.0072 > Toàn bộ: 0.9107 ± 0.0046 > Kích thước: 0.8878 ± 0.0056 |
| F1-macro CV theo bộ đặc trưng (GaussianNB) | Hình dạng: 0.9062 ± 0.0067 > Toàn bộ: 0.7632 ± 0.0085 > Kích thước: 0.7625 ± 0.0080 |
| Tỷ trọng feature_importances_ (DT 16 đặc trưng) | Hình dạng: 59,95% (Top: ShapeFactor1 22,93%, Compactness 18,83%) vs Kích thước: 40,05% |
| Kết luận RQ1 | Nhóm Hình dạng giúp phân loại 7 loại đậu chính xác hơn rõ rệt nhờ đặc tính di truyền học ổn định |
| K tối ưu | |
| ARI | |

## 6. Nhật ký quyết định
| Ngày | Quyết định |
|---|---|
| 2026-11-17 | Giữ nguyên 3.004 mẫu chứa ngoại lai IQR (22,18%) vì là biến thiên sinh học tự nhiên giữa các giống đậu |
| 2026-11-17 | Phân chia 16 đặc trưng thành 6 đặc trưng kích thước (đơn vị đo) và 10 đặc trưng hình dạng (tỷ lệ không thứ nguyên) |
| 2026-11-17 | Chọn mô hình tốt nhất dựa trên F1-macro 5-fold CV trên tập Train: Decision Tree - Hình dạng (8 đặc trưng) |
| 2026-11-17 | Đánh giá mô hình tốt nhất trên tập Test (20%): Accuracy = 0.8963, F1-macro = 0.9099; Bombay đạt 100% |
| 2026-11-19 | Kết luận RQ1: Nhóm Hình dạng vượt trội so với Kích thước trên cả 2 thuật toán (GNB +14,37%, DT +2,50%) và chiếm 59,95% importance |

## 7. Việc cần làm tiếp theo
1. Triển khai **Task T11**: Chuẩn hóa dữ liệu bằng StandardScaler (bỏ nhãn Class), phân cụm K-Means K=2..10 (Elbow và Silhouette) để chọn K tối ưu.
2. Triển khai **Task T12**: Fit K-Means với K tối ưu, tính crosstab và ARI so với nhãn thật, vẽ biểu đồ 10, 11 (PCA 2D) và 12 (heatmap crosstab).
