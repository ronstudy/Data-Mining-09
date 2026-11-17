# PROGRESS - Theo dõi tiến độ

## 1. Tổng quan
| Hạng mục | Giá trị |
|---|---|
| Giai đoạn hiện tại | Đang thực hiện Sprint 2 (EDA và thống kê) |
| Tiến độ | 5/16 task |
| Cập nhật lần cuối | 2026-11-17 |

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
| 2 | T6 | Biểu đồ đa biến 04, 05 | ⬜ | | |
| 3 | T7 | Chia dữ liệu, huấn luyện 6 tổ hợp | ⬜ | | |
| 3 | T8 | 5-fold CV, chọn mô hình | ⬜ | | |
| 3 | T9 | Đánh giá, biểu đồ 06-08 | ⬜ | | |
| 3 | T10 | Kết luận RQ1 | ⬜ | | |
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
| Mô hình tốt nhất (F1-macro CV) | |
| Accuracy / F1-macro trên tập kiểm tra | |
| Nhóm đặc trưng quan trọng hơn | |
| K tối ưu | |
| ARI | |

## 6. Nhật ký quyết định
| Ngày | Quyết định |
|---|---|
| 2026-11-17 | Giữ nguyên 3.004 mẫu chứa ngoại lai IQR (22,18%) vì là biến thiên sinh học tự nhiên giữa các giống đậu |
| 2026-11-17 | Phân chia 16 đặc trưng thành 6 đặc trưng kích thước (đơn vị đo) và 10 đặc trưng hình dạng (tỷ lệ không thứ nguyên) |

## 7. Việc cần làm tiếp theo
1. Triển khai **Task T6**: Biểu đồ đa biến 04 (Heatmap tương quan) và 05 (Boxplot theo lớp), in nhận xét EDA.
2. Chuẩn bị bước vào **Sprint 3: Mô hình hóa** (Tasks T7 - T13).
