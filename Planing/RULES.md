# RULES - Quy tắc dự án

## 1. Quy tắc code sạch
- **PEP8:** thụt lề 4 khoảng trắng, dòng tối đa 88 ký tự, tên hàm và biến dạng `snake_case`, hằng số `UPPER_CASE`.
- **Import:** gom ở đầu file theo thứ tự: thư viện chuẩn, thư viện ngoài. Không `import *`.
- **Hàm:** mỗi bước pipeline là một hàm nhỏ (ví dụ `load_data`, `preprocess`, `run_eda`, `run_classification`, `run_clustering`), gọi từ hàm `main()`. Điểm vào là `if __name__ == "__main__":`.
- **Chú thích:** docstring ngắn bằng tiếng Việt cho mỗi hàm. Comment giải thích lý do (vì sao), không lặp lại code.
- **Hằng số:** không dùng số cố định rải rác. Mọi tham số đặt ở đầu file (`RANDOM_STATE = 42`, `TEST_SIZE = 0.2`, ...).
- **In kết quả:** dùng `print` có tiêu đề từng phần, định dạng số 4 chữ số thập phân.

## 2. Xử lý ngoại lệ
- Kiểm tra trước khi chạy: file CSV tồn tại, đủ 16 cột đặc trưng và cột `Class`.
- Lỗi nghiêm trọng thì dừng bằng `sys.exit(1)` kèm thông báo **tiếng Việt** và hướng khắc phục. Ví dụ: "Không tìm thấy dry_beans.csv. Hãy đặt file cùng thư mục với main.py."
- Chỉ bắt các ngoại lệ cụ thể (`FileNotFoundError`, `KeyError`, `ValueError`). Không dùng `except:` trống.
- Không nuốt lỗi âm thầm: mọi lỗi được bắt đều phải in ra.
- Thư mục `outputs/` tự tạo bằng `os.makedirs(..., exist_ok=True)`.

## 3. Danh mục thư viện cho phép
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, và thư viện chuẩn `os`, `sys`, `warnings`.

## 4. Danh sách AI TUYỆT ĐỐI KHÔNG ĐƯỢC LÀM
1. Tạo thêm file `.py` hoặc chia thành nhiều module. Chỉ có đúng một `main.py`.
2. Tự ý đổi thuật toán hoặc thêm thuật toán ngoài GaussianNB, DecisionTreeClassifier, KMeans.
3. Dùng thư viện ngoài danh mục (không `xgboost`, `scipy` trực tiếp, `plotly`, `tensorflow`...).
4. Xóa hoặc loại bỏ ngoại lai. Chỉ phát hiện bằng IQR và báo cáo.
5. Chuẩn hóa dữ liệu cho Naive Bayes hoặc Decision Tree. `StandardScaler` chỉ dùng cho K-Means và PCA.
6. Đưa cột `Class` vào K-Means.
7. Fit scaler hoặc bất kỳ bước tiền xử lý nào trên toàn bộ dữ liệu rồi mới chia tập (gây rò rỉ dữ liệu).
8. Chọn mô hình dựa trên tập kiểm tra rồi báo kết quả cũng trên đó. Việc chọn mô hình tốt nhất dựa trên F1-macro trung bình của 5-fold CV (trên tập huấn luyện). Tập kiểm tra 20% chỉ dùng để báo cáo kết quả cuối.
9. Thay `RANDOM_STATE = 42`, `TEST_SIZE`, `K_RANGE` mà không được duyệt.
10. Tự bịa số liệu hoặc kết luận chưa có trong output thực tế.
11. Thêm giao diện, web, API, hay lưu mô hình ra file.
12. Tự đổi bảng màu, DPI, hay cấu trúc biểu đồ đã quy định trong `DESIGN.md`.
13. Tải dữ liệu từ mạng trong code.
14. Chỉnh sửa file `.md` trong `Planing/` mà chưa được cho phép.
15. Đánh dấu task hoàn thành khi chưa chạy `main.py` thành công.

## 5. Quy tắc làm việc cho AI
- Trước mỗi thay đổi, đọc `TASKS.md` và `PROGRESS.md`. Sau mỗi task, cập nhật `PROGRESS.md`.
- Mỗi lần chỉ làm một task nhỏ, chạy kiểm tra rồi mới sang task tiếp.
- Khi có xung đột giữa các file tài liệu, hỏi người dùng, không tự quyết.
