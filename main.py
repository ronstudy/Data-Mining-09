"""Dự án khai phá dữ liệu Dry Bean Dataset.

Quy trình phân loại và phân cụm 7 loại đậu khô từ 16 đặc trưng hình thái.
Sprint 1: Chuẩn bị khung chương trình, nạp dữ liệu và tiền xử lý.
Sprint 2: Khám phá dữ liệu (EDA) và phân tích thống kê.
"""

import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==============================================================================
# HẰNG SỐ CẤU HÌNH DỰ ÁN
# ==============================================================================
RANDOM_STATE = 42
DATA_PATH = "dry_beans.csv"
OUTPUT_DIR = "outputs"
TEST_SIZE = 0.2
CV_FOLDS = 5
K_RANGE = range(2, 11)

# Định nghĩa 2 nhóm đặc trưng hình thái
# Nhóm kích thước: các độ đo hình học mang đơn vị độ dài hoặc diện tích (pixel)
SIZE_FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "EquivDiameter",
]

# Nhóm hình dạng: các chỉ số tỷ lệ, độ tròn, độ lồi không thứ nguyên
SHAPE_FEATURES = [
    "AspectRation",
    "Eccentricity",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

ALL_FEATURES = SIZE_FEATURES + SHAPE_FEATURES
CLASS_COLUMN = "Class"
EXPECTED_COLUMNS = ALL_FEATURES + [CLASS_COLUMN]

# Danh sách 7 lớp đậu khô và bảng màu cố định theo tab10 (DESIGN.md)
CLASS_NAMES = [
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
]

TAB10_PALETTE = sns.color_palette("tab10", len(CLASS_NAMES))
CLASS_COLOR_MAP = dict(zip(CLASS_NAMES, TAB10_PALETTE))


# ==============================================================================
# SPRINT 1: NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==============================================================================
def load_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """Đọc tập dữ liệu CSV, kiểm tra cấu trúc cột và báo cáo thông tin cơ bản.

    Tham số:
        file_path: Đường dẫn tới file dữ liệu CSV.

    Trả về:
        pd.DataFrame chứa toàn bộ dữ liệu thô.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 1: NẠP DỮ LIỆU (TASK T2)")
    print("=" * 70)

    # Kiểm tra sự tồn tại của file dữ liệu
    if not os.path.exists(file_path):
        print(
            f"[LỖI] Không tìm thấy file dữ liệu tại đường dẫn: '{file_path}'.\n"
            "Hướng dẫn khắc phục: Vui lòng đảm bảo file 'dry_beans.csv' đã được đặt "
            "đúng tại thư mục gốc của dự án, cùng cấp với 'main.py'."
        )
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(
            f"[LỖI] Không thể đọc file CSV '{file_path}': {exc}\n"
            "Hướng dẫn khắc phục: Kiểm tra quyền đọc file hoặc định dạng file CSV."
        )
        sys.exit(1)

    # Kiểm tra số lượng và tên các cột bắt buộc
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        print(
            f"[LỖI] Dữ liệu thiếu các cột bắt buộc: {missing_columns}.\n"
            f"Tổng số cột hiện có: {len(df.columns)}/{len(EXPECTED_COLUMNS)} cột.\n"
            "Hướng dẫn khắc phục: Vui lòng kiểm tra lại cấu trúc file dữ liệu để "
            "đảm bảo có đủ 16 đặc trưng hình thái và 1 cột nhãn 'Class'."
        )
        sys.exit(1)

    print(f"- Đọc thành công file: '{file_path}'")
    print(f"- Kích thước tập dữ liệu thô: {df.shape[0]} dòng, {df.shape[1]} cột")

    # In thông tin kiểu dữ liệu của các cột
    print("\nKiểu dữ liệu các cột:")
    for col in df.columns:
        print(f"  + {col:<18}: {df[col].dtype}")

    # Báo cáo phân bố số lượng và tỷ lệ của 7 lớp
    print("\nPhân bố số lượng các lớp mục tiêu ('Class'):")
    class_counts = df[CLASS_COLUMN].value_counts()
    class_ratios = df[CLASS_COLUMN].value_counts(normalize=True) * 100
    for cls in class_counts.index:
        count = class_counts[cls]
        ratio = class_ratios[cls]
        print(f"  + {cls:<10}: {count:>5} mẫu ({ratio:>6.2f}%)")

    return df


def preprocess(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Tiền xử lý dữ liệu: kiểm tra thiếu, loại trùng và thống kê ngoại lai IQR.

    Tuyệt đối không xóa dữ liệu ngoại lai và không chuẩn hóa ở bước này.

    Tham số:
        df: DataFrame thô sau khi nạp.

    Trả về:
        Tuple gồm (df_clean, SIZE_FEATURES, SHAPE_FEATURES).
    """
    print("\n" + "=" * 70)
    print("BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU (TASK T3)")
    print("=" * 70)

    # 1. Kiểm tra giá trị thiếu (Missing values)
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"- Kiểm tra giá trị khuyết thiếu (NaN): {total_nulls} giá trị thiếu.")
    if total_nulls > 0:
        print("  Chi tiết các cột có giá trị thiếu:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  + {col}: {count} dòng thiếu")
    else:
        print("  -> Dữ liệu hoàn toàn đầy đủ, không có giá trị thiếu.")

    # 2. Kiểm tra và loại bỏ các dòng trùng lặp hoàn toàn
    num_duplicates = df.duplicated().sum()
    print(f"\n- Kiểm tra dòng trùng lặp: Phát hiện {num_duplicates} dòng trùng.")
    if num_duplicates > 0:
        df_clean = df.drop_duplicates().copy().reset_index(drop=True)
        print(
            f"  -> Đã loại bỏ {num_duplicates} dòng trùng lặp hoàn toàn.\n"
            f"  -> Kích thước dữ liệu sau khi loại trùng: "
            f"{df_clean.shape[0]} dòng, {df_clean.shape[1]} cột."
        )
    else:
        df_clean = df.copy()
        print("  -> Không có dòng trùng lặp nào.")

    # 3. Thống kê và phát hiện ngoại lai bằng phương pháp IQR (giữ nguyên ngoại lai)
    print("\n- Báo cáo dữ liệu ngoại lai theo phương pháp IQR (ngưỡng 1.5 * IQR):")
    outlier_summary = {}
    any_outlier_mask = pd.Series(False, index=df_clean.index)

    for feature in ALL_FEATURES:
        q1 = df_clean[feature].quantile(0.25)
        q3 = df_clean[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        is_outlier = (df_clean[feature] < lower_bound) | (
            df_clean[feature] > upper_bound
        )
        count_outliers = int(is_outlier.sum())
        outlier_summary[feature] = count_outliers
        any_outlier_mask |= is_outlier

        ratio = (count_outliers / len(df_clean)) * 100
        print(
            f"  + {feature:<16}: {count_outliers:>4} ngoại lai ({ratio:>5.2f}%) "
            f"[Khoảng hợp lệ: {lower_bound:.4f} đến {upper_bound:.4f}]"
        )

    total_rows_with_outlier = int(any_outlier_mask.sum())
    total_outlier_ratio = (total_rows_with_outlier / len(df_clean)) * 100
    print(
        f"\n  => Tổng số mẫu có ít nhất một đặc trưng ngoại lai: "
        f"{total_rows_with_outlier}/{len(df_clean)} mẫu "
        f"({total_outlier_ratio:.2f}%)."
    )
    print(
        "  => QUYẾT ĐỊNH XỬ LÝ: Tuyệt đối giữ nguyên toàn bộ các mẫu ngoại lai.\n"
        "     Lý do: Các giá trị này phản ánh sự biến thiên tự nhiên về hình thái\n"
        "     của từng giống đậu (ví dụ giống Bombay có kích thước lớn tự nhiên),\n"
        "     không phải lỗi đo lường hay hư hỏng dữ liệu."
    )

    # 4. Xác nhận phân tách 2 nhóm đặc trưng
    print("\n- Phân tách nhóm đặc trưng phục vụ phân tích câu hỏi nghiên cứu (RQ1):")
    print(f"  + Nhóm kích thước ({len(SIZE_FEATURES)} đặc trưng): {SIZE_FEATURES}")
    print(f"  + Nhóm hình dạng  ({len(SHAPE_FEATURES)} đặc trưng): {SHAPE_FEATURES}")

    return df_clean, SIZE_FEATURES, SHAPE_FEATURES


# ==============================================================================
# SPRINT 2: KHÁM PHÁ DỮ LIỆU (EDA) VÀ THỐNG KÊ
# ==============================================================================
def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán và in bảng thống kê mô tả cho 16 đặc trưng số.

    Tham số:
        df: DataFrame đã làm sạch.

    Trả về:
        DataFrame chứa các chỉ số thống kê mô tả (mean, median, std, Q1, Q3).
    """
    print("\n" + "=" * 70)
    print("BƯỚC 3: KHÁM PHÁ DỮ LIỆU - EDA (TASK T4)")
    print("=" * 70)
    print("\n1. BẢNG THỐNG KÊ MÔ TẢ (16 ĐẶC TRƯNG SỐ):")

    stats_list = []
    for col in ALL_FEATURES:
        series = df[col]
        q1 = series.quantile(0.25)
        median = series.median()
        q3 = series.quantile(0.75)
        stats_list.append(
            {
                "Đặc trưng": col,
                "Mean": series.mean(),
                "Std": series.std(),
                "Median": median,
                "Q1 (25%)": q1,
                "Q3 (75%)": q3,
                "IQR": q3 - q1,
                "Min": series.min(),
                "Max": series.max(),
            }
        )

    stats_df = pd.DataFrame(stats_list)

    # Định dạng bảng in console với 4 chữ số thập phân (RULES.md)
    header = (
        f"{'Đặc trưng':<18} | {'Mean':>11} | {'Std':>11} | "
        f"{'Median':>11} | {'Q1 (25%)':>11} | {'Q3 (75%)':>11}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for _, row in stats_df.iterrows():
        print(
            f"{row['Đặc trưng']:<18} | "
            f"{row['Mean']:>11.4f} | "
            f"{row['Std']:>11.4f} | "
            f"{row['Median']:>11.4f} | "
            f"{row['Q1 (25%)']:>11.4f} | "
            f"{row['Q3 (75%)']:>11.4f}"
        )
    print("-" * len(header))

    return stats_df


def plot_class_distribution(
    df: pd.DataFrame, output_dir: str = OUTPUT_DIR
) -> None:
    """Vẽ biểu đồ cột phân bố 7 lớp hạt đậu và lưu vào outputs/01_phan_bo_lop.png.

    Tham số:
        df: DataFrame đã làm sạch.
        output_dir: Thư mục lưu file biểu đồ.
    """
    print("\n2. TRỰC QUAN HÓA BIỂU ĐỒ 01: PHÂN BỐ LỚP (TASK T4)")

    # Thiết lập style và font theo tiêu chuẩn DESIGN.md
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    # Lấy số lượng từng lớp theo thứ tự gán màu chuẩn tab10 (DESIGN.md)
    class_counts = df[CLASS_COLUMN].value_counts()
    counts = [class_counts.get(cls, 0) for cls in CLASS_NAMES]
    colors = [CLASS_COLOR_MAP[cls] for cls in CLASS_NAMES]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        CLASS_NAMES,
        counts,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )

    # Hiển thị số lượng mẫu cụ thể trên đầu mỗi cột
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=10, fontweight="bold")

    # Định dạng tiêu đề và nhãn trục tiếng Việt (DESIGN.md)
    ax.set_title(
        "Phân bố số lượng các lớp hạt đậu (Dry Bean Dataset)",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Loại đậu (Class)", fontsize=11, labelpad=8)
    ax.set_ylabel("Số lượng mẫu", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Mở rộng giới hạn trục Y để nhãn số lượng không chạm viền trên
    ax.set_ylim(0, max(counts) * 1.12)
    plt.tight_layout()

    # Xuất ảnh DPI 300, bbox_inches='tight' theo quy định
    output_path = os.path.join(output_dir, "01_phan_bo_lop.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"- Đã lưu biểu đồ thành công: '{output_path}'")

    plt.show()
    plt.close(fig)


def run_task_t4(
    df: pd.DataFrame, output_dir: str = OUTPUT_DIR
) -> pd.DataFrame:
    """Thực thi toàn bộ Task T4: Thống kê mô tả và biểu đồ phân bố lớp.

    Tham số:
        df: DataFrame đã làm sạch sau Bước 2.
        output_dir: Thư mục lưu biểu đồ đầu ra.

    Trả về:
        DataFrame chứa bảng thống kê mô tả 16 đặc trưng số.
    """
    stats_df = compute_descriptive_statistics(df)
    plot_class_distribution(df, output_dir)
    return stats_df


def plot_histograms_kde(
    df: pd.DataFrame, output_dir: str = OUTPUT_DIR
) -> None:
    """Vẽ lưới 4x4 biểu đồ Histogram kèm đường KDE cho 16 đặc trưng số.

    Tham số:
        df: DataFrame đã làm sạch.
        output_dir: Thư mục lưu file biểu đồ.
    """
    print("\n1. TRỰC QUAN HÓA BIỂU ĐỒ 02: HISTOGRAM & KDE (TASK T5)")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(ALL_FEATURES):
        ax = axes[idx]
        sns.histplot(
            df[feature],
            kde=True,
            ax=ax,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.6,
        )
        ax.set_title(feature, fontsize=12, fontweight="bold")
        ax.set_xlabel("Giá trị", fontsize=11)
        ax.set_ylabel("Số lượng", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(
        "Phân bố đơn biến 16 đặc trưng hình thái (Histogram & KDE)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "02_histogram_kde.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"- Đã lưu biểu đồ thành công: '{output_path}'")

    plt.show()
    plt.close(fig)


def plot_boxplots(
    df: pd.DataFrame, output_dir: str = OUTPUT_DIR
) -> None:
    """Vẽ lưới 4x4 biểu đồ Boxplot cho 16 đặc trưng số để quan sát ngoại lai.

    Tham số:
        df: DataFrame đã làm sạch.
        output_dir: Thư mục lưu file biểu đồ.
    """
    print("\n2. TRỰC QUAN HÓA BIỂU ĐỒ 03: BOXPLOT (TASK T5)")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature in enumerate(ALL_FEATURES):
        ax = axes[idx]
        sns.boxplot(
            y=df[feature],
            ax=ax,
            color="#2ca02c",
            width=0.4,
            fliersize=3,
            flierprops=dict(
                marker="o",
                markerfacecolor="crimson",
                markeredgecolor="none",
                alpha=0.4,
            ),
        )
        ax.set_title(feature, fontsize=12, fontweight="bold")
        ax.set_ylabel("Giá trị", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle(
        "Biểu đồ hộp (Boxplot) quan sát ngoại lai cho 16 đặc trưng hình thái",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "03_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"- Đã lưu biểu đồ thành công: '{output_path}'")

    plt.show()
    plt.close(fig)


def run_task_t5(
    df: pd.DataFrame, output_dir: str = OUTPUT_DIR
) -> None:
    """Thực thi toàn bộ Task T5: Trực quan hóa đơn biến (Biểu đồ 02 & 03).

    Tham số:
        df: DataFrame đã làm sạch sau Bước 2.
        output_dir: Thư mục lưu biểu đồ đầu ra.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 4: TRỰC QUAN HÓA ĐƠN BIẾN (TASK T5)")
    print("=" * 70)
    plot_histograms_kde(df, output_dir)
    plot_boxplots(df, output_dir)


# ==============================================================================
# ĐIỂM VÀO CHÍNH CỦA CHƯƠNG TRÌNH
# ==============================================================================
def main() -> None:
    """Hàm điều phối chính cho toàn bộ quy trình khai phá dữ liệu."""
    # Bỏ qua các cảnh báo không ảnh hưởng kết quả thực thi
    warnings.filterwarnings("ignore")

    print("\n" + "#" * 70)
    print("# DỰ ÁN KHAI PHÁ DỮ LIỆU: PHÂN LOẠI ĐẬU KHÔ (DRY BEAN DATASET)")
    print("# TRIỂN KHAI SPRINT 1 & 2: TIỀN XỬ LÝ VÀ EDA (TASKS T1 - T5)")
    print("#" * 70)

    # Tự động tạo thư mục chứa ảnh đầu ra theo quy chuẩn ARCHITECTURE.md
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Task T2: Nạp dữ liệu
    df_raw = load_data(DATA_PATH)

    # Task T3: Tiền xử lý dữ liệu
    df_clean, size_features, shape_features = preprocess(df_raw)

    # Task T4: Thống kê mô tả và trực quan hóa phân bố lớp
    stats_df = run_task_t4(df_clean, OUTPUT_DIR)

    # Task T5: Trực quan hóa đơn biến (Histogram + KDE và Boxplot)
    run_task_t5(df_clean, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH TASKS T4 & T5 (SPRINT 2) THÀNH CÔNG!")
    print(f"- Đã tính toán thống kê mô tả cho {len(stats_df)} đặc trưng.")
    print("- Đã xuất 3 biểu đồ: 01_phan_bo_lop.png, 02_histogram_kde.png,")
    print("  03_boxplot.png (DPI 300) vào thư mục outputs/.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
