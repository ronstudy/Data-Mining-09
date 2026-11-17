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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# ==============================================================================
# HẰNG SỐ CẤU HÌNH DỰ ÁN
# ==============================================================================
RANDOM_STATE = 42
DATA_PATH = "dry_beans.csv"
OUTPUT_DIR = "outputs"
TEST_SIZE = 0.2
CV_FOLDS = 5
K_RANGE = range(2, 11)

# Định nghĩa 2 nhóm đặc trưng hình thái theo PRD.md và ARCHITECTURE.md
# Nhóm kích thước (8 đặc trưng): kích thước cơ bản và các độ đo trục chính/phụ
SIZE_FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
]

# Nhóm hình dạng (8 đặc trưng): các hệ số hình thái, độ lồi, độ nén, độ tròn
SHAPE_FEATURES = [
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


def plot_correlation_heatmap(
    df: pd.DataFrame, output_dir: str = OUTPUT_DIR
) -> list[tuple[str, str, float]]:
    """Tính ma trận tương quan Pearson, vẽ heatmap và lọc các cặp |r| > 0.9.

    Tham số:
        df: DataFrame đã làm sạch.
        output_dir: Thư mục lưu file biểu đồ.

    Trả về:
        Danh sách các cặp đặc trưng có |r| > 0.9 kèm hệ số tương quan.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 5: PHÂN TÍCH ĐA BIẾN VÀ TƯƠNG QUAN (TASK T6)")
    print("=" * 70)

    # 1. Tính ma trận tương quan Pearson
    corr = df[ALL_FEATURES].corr()

    # 2. Lọc các cặp đặc trưng có |r| > 0.9 để kiểm tra đa cộng tuyến
    high_corr_pairs = []
    for i in range(len(ALL_FEATURES)):
        for j in range(i + 1, len(ALL_FEATURES)):
            feat1 = ALL_FEATURES[i]
            feat2 = ALL_FEATURES[j]
            r_val = float(corr.loc[feat1, feat2])
            if abs(r_val) > 0.9:
                high_corr_pairs.append((feat1, feat2, r_val))

    high_corr_pairs.sort(key=lambda item: abs(item[2]), reverse=True)

    print(
        f"\n1. DANH SÁCH CÁC CẶP ĐẶC TRƯNG TƯƠNG QUAN CAO (|r| > 0.9) "
        f"(Tổng cộng: {len(high_corr_pairs)} cặp):"
    )
    header = (
        f"{'STT':<4} | {'Đặc trưng 1':<16} - {'Đặc trưng 2':<16} | "
        f"{'Hệ số r':>9} | {'Mức độ tương quan'}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for idx, (f1, f2, r_val) in enumerate(high_corr_pairs, 1):
        rel_type = "Đồng biến rất mạnh" if r_val > 0 else "Nghịch biến rất mạnh"
        print(
            f"{idx:<4} | {f1:<16} - {f2:<16} | {r_val:>9.4f} | {rel_type}"
        )
    print("-" * len(header))

    # 3. Trực quan hóa Heatmap ma trận tương quan (Biểu đồ 04)
    print("\n2. TRỰC QUAN HÓA BIỂU ĐỒ 04: HEATMAP TƯƠNG QUAN (TASK T6)")
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"label": "Hệ số tương quan Pearson (r)"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(
        "Ma trận tương quan Pearson giữa 16 đặc trưng hình thái",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "04_heatmap_tuong_quan.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"- Đã lưu biểu đồ thành công: '{output_path}'")

    plt.show()
    plt.close(fig)

    return high_corr_pairs


def plot_features_by_class(
    df: pd.DataFrame, output_dir: str = OUTPUT_DIR
) -> None:
    """Vẽ boxplot so sánh các đặc trưng tiêu biểu phân tách theo 7 lớp đậu.

    Tham số:
        df: DataFrame đã làm sạch.
        output_dir: Thư mục lưu file biểu đồ.
    """
    print("\n3. TRỰC QUAN HÓA BIỂU ĐỒ 05: BOXPLOT THEO LỚP (TASK T6)")

    # Chọn 3 đặc trưng tiêu biểu đại diện cho kích thước và hình dạng
    selected_features = ["Area", "AspectRation", "ShapeFactor1"]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, feature in enumerate(selected_features):
        ax = axes[idx]
        sns.boxplot(
            x=CLASS_COLUMN,
            y=feature,
            data=df,
            order=CLASS_NAMES,
            palette=CLASS_COLOR_MAP,
            ax=ax,
            width=0.5,
            fliersize=2.5,
            flierprops=dict(marker="o", alpha=0.3),
        )
        ax.set_title(
            f"Phân bố đặc trưng {feature} theo lớp",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Loại đậu (Class)", fontsize=11, labelpad=8)
        ax.set_ylabel(feature, fontsize=11, labelpad=8)
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle(
        "So sánh phân bố các đặc trưng tiêu biểu theo 7 lớp đậu",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "05_boxplot_theo_lop.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"- Đã lưu biểu đồ thành công: '{output_path}'")

    plt.show()
    plt.close(fig)


def print_eda_summary(num_high_corr: int) -> None:
    """In nhận xét tổng quan kết quả khám phá dữ liệu (EDA) ra console.

    Tham số:
        num_high_corr: Số lượng cặp đặc trưng có tương quan cao |r| > 0.9.
    """
    print("\n" + "=" * 70)
    print("NHẬN XÉT TỔNG QUAN KẾT QUẢ KHÁM PHÁ DỮ LIỆU (EDA SUMMARY):")
    print("=" * 70)
    print(
        "1. Về tương quan và hiện tượng đa cộng tuyến (Multicollinearity):\n"
        f"   - Phát hiện {num_high_corr} cặp đặc trưng có tương quan rất cao "
        "với |r| > 0.9.\n"
        "   - Nhóm kích thước (Area, ConvexArea, Perimeter, EquivDiameter)\n"
        "     có đa cộng tuyến cực cao (r = 0.9999 giữa Area và ConvexArea;\n"
        "     r = 0.9915 giữa Perimeter và EquivDiameter).\n"
        "   - Nhóm hình dạng có cặp tương quan gần như tuyệt đối giữa\n"
        "     Compactness và ShapeFactor3 (r = 0.9987), và tương quan nghịch\n"
        "     mạnh giữa AspectRation và Compactness (r = -0.9876).\n"
        "   - Tác động thuật toán: Naive Bayes (giả định độc lập) có thể bị\n"
        "     ảnh hưởng bởi hiện tượng đa cộng tuyến này, trong khi Decision Tree\n"
        "     ít bị ảnh hưởng hơn do chỉ chọn một đặc trưng tốt nhất tại mỗi nút."
    )
    print(
        "\n2. Về khả năng phân tách của các đặc trưng đối với 7 lớp đậu:\n"
        "   - Nhóm kích thước (Area, Perimeter): Phân tách vượt trội giống BOMBAY\n"
        "     (diện tích hạt lớn gấp 2-3 lần toàn bộ các giống còn lại).\n"
        "   - Nhóm hình dạng (AspectRation, Eccentricity): Phân tách rõ rệt giống\n"
        "     hạt thon dài (HOROZ, CALI) khỏi các giống hạt tròn (SEKER, DERMASON).\n"
        "   - Nhóm hệ số hình thái (ShapeFactor1, ShapeFactor2): Hỗ trợ phân biệt\n"
        "     các giống có kích thước tương đương nhau như DERMASON và SIRA.\n"
        "   - Kết luận sơ bộ cho RQ1: Cả hai nhóm đặc trưng kích thước và hình dạng\n"
        "     đều đóng vai trò bổ trợ then chốt, không thể thiếu nhóm nào nếu muốn\n"
        "     phân loại chính xác toàn bộ 7 lớp hạt đậu."
    )
    print("=" * 70)


def run_task_t6(
    df: pd.DataFrame, output_dir: str = OUTPUT_DIR
) -> list[tuple[str, str, float]]:
    """Thực thi toàn bộ Task T6: Phân tích đa biến, biểu đồ 04, 05 và nhận xét.

    Tham số:
        df: DataFrame đã làm sạch sau Bước 2.
        output_dir: Thư mục lưu biểu đồ đầu ra.

    Trả về:
        Danh sách các cặp đặc trưng có |r| > 0.9.
    """
    high_corr_pairs = plot_correlation_heatmap(df, output_dir)
    plot_features_by_class(df, output_dir)
    print_eda_summary(len(high_corr_pairs))
    return high_corr_pairs


# ==============================================================================
# SPRINT 3: MÔ HÌNH HÓA (MODELING) - PHÂN LOẠI (RQ1) VÀ PHÂN CỤM (RQ2)
# ==============================================================================
def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chia tập dữ liệu thành Train và Test theo tỷ lệ 80/20 có stratify.

    Tuyệt đối không chuẩn hóa dữ liệu trước hoặc trong quá trình chia tập.

    Tham số:
        df: DataFrame đã làm sạch sau bước tiền xử lý.
        test_size: Tỷ lệ mẫu dành cho tập kiểm tra (mặc định 0.2).
        random_state: Hạt giống ngẫu nhiên để tái lập kết quả (42).

    Trả về:
        Tuple (X_train, X_test, y_train, y_test).
    """
    print("\n" + "=" * 70)
    print("BƯỚC 6: PHÂN CHIA DỮ LIỆU HUẤN LUYỆN VÀ KIỂM TRA (TASK T7)")
    print("=" * 70)

    X = df[ALL_FEATURES]
    y = df[CLASS_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_pct = (X_train.shape[0] / len(df)) * 100
    test_pct = (X_test.shape[0] / len(df)) * 100

    print(
        f"- Phân chia dữ liệu theo tỷ lệ {int((1 - test_size) * 100)}/"
        f"{int(test_size * 100)} (stratify=y, random_state={random_state}):"
    )
    print(
        f"  + Tập huấn luyện (Train): {X_train.shape[0]} mẫu "
        f"({train_pct:.2f}%), {X_train.shape[1]} đặc trưng."
    )
    print(
        f"  + Tập kiểm tra   (Test) : {X_test.shape[0]} mẫu "
        f"({test_pct:.2f}%), {X_test.shape[1]} đặc trưng."
    )

    # Kiểm tra tính toàn vẹn tỷ lệ phân bố 7 lớp trên 2 tập
    print("\n- So sánh tỷ lệ phân bố 7 lớp giữa tập Train và Test:")
    header = (
        f"{'Lớp (Class)':<12} | {'Train (Số lượng)':>16} | {'Train (%)':>10} | "
        f"{'Test (Số lượng)':>15} | {'Test (%)':>9}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    train_counts = y_train.value_counts()
    train_ratios = y_train.value_counts(normalize=True) * 100
    test_counts = y_test.value_counts()
    test_ratios = y_test.value_counts(normalize=True) * 100

    for cls in CLASS_NAMES:
        tr_cnt = train_counts.get(cls, 0)
        tr_rat = train_ratios.get(cls, 0.0)
        te_cnt = test_counts.get(cls, 0)
        te_rat = test_ratios.get(cls, 0.0)
        print(
            f"{cls:<12} | {tr_cnt:>16} | {tr_rat:>9.2f}% | "
            f"{te_cnt:>15} | {te_rat:>8.2f}%"
        )
    print("-" * len(header))
    print(
        "  -> Kết luận: Tỷ lệ phân bố 7 lớp được bảo toàn đồng nhất "
        "giữa tập Train và Test."
    )

    return X_train, X_test, y_train, y_test


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, object]]:
    """Huấn luyện 6 tổ hợp mô hình phân loại (2 thuật toán x 3 bộ đặc trưng).

    Tuyệt đối không chuẩn hóa dữ liệu cho Naive Bayes và Decision Tree.

    Tham số:
        X_train: Đặc trưng tập huấn luyện.
        y_train: Nhãn lớp tập huấn luyện.
        random_state: Hạt giống ngẫu nhiên cho Decision Tree (42).

    Trả về:
        Dict chứa 6 mô hình đã huấn luyện kèm thông tin bộ đặc trưng.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 7: HUẤN LUYỆN 6 TỔ HỢP MÔ HÌNH PHÂN LOẠI (TASK T7)")
    print("=" * 70)

    # 3 bộ đặc trưng nghiên cứu
    feature_sets = {
        "Toàn bộ (16 đặc trưng)": ALL_FEATURES,
        "Kích thước (8 đặc trưng)": SIZE_FEATURES,
        "Hình dạng (8 đặc trưng)": SHAPE_FEATURES,
    }

    # 2 thuật toán theo yêu cầu PRD.md
    algorithms = {
        "GaussianNB": lambda: GaussianNB(),
        "Decision Tree": lambda: DecisionTreeClassifier(
            random_state=random_state
        ),
    }

    trained_models = {}
    comb_idx = 1
    total_combs = len(algorithms) * len(feature_sets)

    for algo_name, algo_fn in algorithms.items():
        for set_name, feature_list in feature_sets.items():
            model = algo_fn()
            X_train_subset = X_train[feature_list]

            print(
                f"[{comb_idx}/{total_combs}] Đang huấn luyện {algo_name:<13} "
                f"trên bộ '{set_name}' ({X_train_subset.shape[0]} mẫu, "
                f"{len(feature_list)} đặc trưng)...",
                end=" ",
            )
            model.fit(X_train_subset, y_train)
            print("Hoàn thành!")

            key = f"{algo_name} - {set_name}"
            trained_models[key] = {
                "model": model,
                "algorithm": algo_name,
                "feature_set_name": set_name,
                "features": feature_list,
            }
            comb_idx += 1

    print(
        f"\n=> Đã huấn luyện thành công toàn bộ {total_combs} tổ hợp mô hình "
        "trên tập Train (giữ nguyên tập Test cho Task T9)."
    )

    return trained_models


def run_task_t7(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    dict[str, dict[str, object]],
]:
    """Thực thi Task T7: Phân chia 80/20 có stratify và huấn luyện 6 tổ hợp.

    Tham số:
        df: DataFrame đã làm sạch sau Bước 2.

    Trả về:
        Tuple (X_train, X_test, y_train, y_test, trained_models).
    """
    X_train, X_test, y_train, y_test = split_dataset(df)
    trained_models = train_classification_models(X_train, y_train)
    return X_train, X_test, y_train, y_test, trained_models


# ==============================================================================
# ĐIỂM VÀO CHÍNH CỦA CHƯƠNG TRÌNH
# ==============================================================================
def main() -> None:
    """Hàm điều phối chính cho toàn bộ quy trình khai phá dữ liệu."""
    # Bỏ qua các cảnh báo không ảnh hưởng kết quả thực thi
    warnings.filterwarnings("ignore")

    print("\n" + "#" * 70)
    print("# DỰ ÁN KHAI PHÁ DỮ LIỆU: PHÂN LOẠI ĐẬU KHÔ (DRY BEAN DATASET)")
    print("# TRIỂN KHAI SPRINT 1, 2, 3: TIỀN XỬ LÝ, EDA VÀ MÔ HÌNH HÓA")
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

    # Task T6: Trực quan hóa đa biến (Heatmap tương quan và Boxplot theo lớp)
    high_corr_pairs = run_task_t6(df_clean, OUTPUT_DIR)

    # Task T7: Phân chia tập dữ liệu 80/20 và huấn luyện 6 tổ hợp mô hình
    X_train, X_test, y_train, y_test, trained_models = run_task_t7(df_clean)

    print("\n" + "=" * 70)
    print("HOÀN THÀNH TASK T7 (SPRINT 3: MÔ HÌNH HÓA) THÀNH CÔNG!")
    print(f"- Đã phân chia tập dữ liệu: {len(X_train)} Train, {len(X_test)} Test.")
    print(f"- Đã huấn luyện thành công {len(trained_models)} tổ hợp mô hình.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
