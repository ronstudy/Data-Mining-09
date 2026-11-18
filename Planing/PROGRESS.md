# PROGRESS - Theo dõi tiến độ

## 1. Tổng quan
| Hạng mục | Giá trị |
|---|---|
| Giai đoạn hiện tại | Đã hoàn thành Sprint 3 (Chuẩn bị Sprint 4: Nghiệm thu) |
| Tiến độ | 13/16 task |
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
| 3 | T11 | K-Means, Elbow, Silhouette | ✅ | 2026-11-19 | Chạy KMeans K=2..10 trên 13.543 mẫu chuẩn hóa, xuất 09_elbow_silhouette.png; Sil cao nhất tại K=3 (0.4044), tại K=7 là 0.3088 |
| 3 | T12 | Crosstab, ARI, biểu đồ 10-12 | ✅ | 2026-11-19 | KMeans K=7, ARI = 0.6669, PCA 2D giải thích 81.91% phương sai; xuất ảnh 10 (crosstab), 11 (PCA cụm), 12 (PCA nhãn thật) |
| 3 | T13 | Kết luận RQ2 | ✅ | 2026-11-19 | K tối ưu K=3 (Silhouette=0.4044), tại K=7 đạt ARI=0.6669; K-Means tách sạch Bombay (100%), Horoz, Seker, Dermason; gom chung Cali/Barbunya |
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
| K tối ưu (Silhouette) | K = 3 (Silhouette = 0.4044; Inertia = 94.331,7346) |
| Chỉ số tại K = 7 (số lớp thực tế) | Silhouette = 0.3088; Inertia = 48.671,4570 |
| Tốc độ suy giảm Inertia (Elbow) | Phẳng dần từ K = 3 đến K = 5 (tụ về 3 nhóm hình thái tự nhiên) |
| ARI (K=7 K-Means so với 7 lớp thực tế) | 0.6669 (mức độ tương đồng phân cụm và phân loại khá cao) |
| Độ thuần nhất cụm K-Means (K=7) | Cụm 3: BOMBAY (100% thuần nhất); Cụm 0: DERMASON (92,21%); Cụm 2: HOROZ (93,83%); Cụm 5: SEKER (92,18%); Cụm 6: SIRA (74,71%); Cụm 4 gom chung CALI và BARBUNYA |
| Tỷ lệ phương sai giải thích PCA 2D | PC1 = 55,55%, PC2 = 26,36% (Tổng = 81,91% phương sai toàn bộ 16 đặc trưng) |
| Kết luận RQ2 | K tối ưu là K=3 (Silhouette=0.4044); K-Means K=7 đạt ARI = 0.6669 khớp rất tốt với phân loại sinh học, tái tạo thuần khiết 4/7 giống đậu |

## 6. Nhật ký quyết định
| Ngày | Quyết định |
|---|---|
| 2026-11-17 | Giữ nguyên 3.004 mẫu chứa ngoại lai IQR (22,18%) vì là biến thiên sinh học tự nhiên giữa các giống đậu |
| 2026-11-17 | Phân chia 16 đặc trưng thành 6 đặc trưng kích thước (đơn vị đo) và 10 đặc trưng hình dạng (tỷ lệ không thứ nguyên) |
| 2026-11-17 | Chọn mô hình tốt nhất dựa trên F1-macro 5-fold CV trên tập Train: Decision Tree - Hình dạng (8 đặc trưng) |
| 2026-11-17 | Đánh giá mô hình tốt nhất trên tập Test (20%): Accuracy = 0.8963, F1-macro = 0.9099; Bombay đạt 100% |
| 2026-11-19 | Kết luận RQ1: Nhóm Hình dạng vượt trội so với Kích thước trên cả 2 thuật toán (GNB +14,37%, DT +2,50%) và chiếm 59,95% importance |
| 2026-11-19 | Hoàn thành T11: Phân cụm K-Means K=2..10 trên 13.543 mẫu chuẩn hóa, xác định K=3 đạt Silhouette cực đại (0.4044), Elbow phẳng từ K=3..5 |
| 2026-11-19 | Hoàn thành T12: Chạy KMeans K=7, lập crosstab, tính ARI = 0.6669, phân tích PCA 2D (81.91% phương sai), xuất 10_crosstab_cum_lop.png, 11_pca_cum.png, 12_pca_lop_that.png |
| 2026-11-19 | Hoàn thành T13: Trả lời câu hỏi RQ2, kết luận K tối ưu K=3 (Sil=0.4044) và độ khớp với 7 loại đậu thực tế ARI = 0.6669 (Sprint 3 hoàn tất 100%) |

## 7. Việc cần làm tiếp theo
1. Triển khai **Sprint 4 – Task T14**: Chạy `main.py` hai lần liên tiếp để kiểm tra tính tái lập (reproducibility) của toàn bộ quy trình từ T1 đến T13.
2. Triển khai **Sprint 4 – Task T15**: Rà soát tuân thủ PEP8 và `RULES.md`, kiểm tra đủ 12 file biểu đồ chuẩn DPI 300 trong thư mục `outputs/`.
3. Triển khai **Sprint 4 – Task T16**: Đối chiếu tiêu chí nghiệm thu trong `PRD.md` và chuẩn bị số liệu hoàn chỉnh cho báo cáo 7 phần.
