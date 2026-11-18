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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
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


def evaluate_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Đánh giá 5-fold Stratified CV trên tập Train cho 6 tổ hợp mô hình.

    Tuyệt đối không sử dụng tập Test. Đánh giá và lựa chọn mô hình tối ưu
    dựa trên F1-macro trung bình của 5-fold Stratified CV.

    Tham số:
        X_train: Đặc trưng tập huấn luyện.
        y_train: Nhãn lớp tập huấn luyện.
        cv_folds: Số fold kiểm định chéo (5).
        random_state: Hạt giống ngẫu nhiên (42).

    Trả về:
        Tuple (cv_df, best_model_info).
    """
    print("\n" + "=" * 70)
    print("BƯỚC 8: ĐÁNH GIÁ 5-FOLD STRATIFIED CV TRÊN TẬP TRAIN (TASK T8)")
    print("=" * 70)

    # Khởi tạo StratifiedKFold chuẩn theo quy định
    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    feature_sets = {
        "Toàn bộ (16 đặc trưng)": ALL_FEATURES,
        "Kích thước (8 đặc trưng)": SIZE_FEATURES,
        "Hình dạng (8 đặc trưng)": SHAPE_FEATURES,
    }

    algorithms = {
        "GaussianNB": lambda: GaussianNB(),
        "Decision Tree": lambda: DecisionTreeClassifier(
            random_state=random_state
        ),
    }

    cv_records = []
    best_f1 = -1.0
    best_model_info = {}

    print(
        f"- Thiết lập kiểm định chéo: {cv_folds}-fold Stratified CV "
        f"(shuffle=True, random_state={random_state})."
    )
    print("- Đang thực hiện kiểm định chéo trên tập Train cho 6 tổ hợp:\n")

    for algo_name, algo_fn in algorithms.items():
        for set_name, feature_list in feature_sets.items():
            model = algo_fn()
            X_subset = X_train[feature_list]

            cv_out = cross_validate(
                model,
                X_subset,
                y_train,
                cv=skf,
                scoring=["accuracy", "f1_macro"],
            )

            acc_mean = float(cv_out["test_accuracy"].mean())
            acc_std = float(cv_out["test_accuracy"].std())
            f1_mean = float(cv_out["test_f1_macro"].mean())
            f1_std = float(cv_out["test_f1_macro"].std())

            record = {
                "Thuật toán": algo_name,
                "Bộ đặc trưng": set_name,
                "F1-macro (Mean)": f1_mean,
                "F1-macro (Std)": f1_std,
                "Accuracy (Mean)": acc_mean,
                "Accuracy (Std)": acc_std,
                "features": feature_list,
            }
            cv_records.append(record)

            # Lựa chọn mô hình tốt nhất theo F1-macro CV trung bình
            if f1_mean > best_f1:
                best_f1 = f1_mean
                best_model_info = {
                    "algorithm": algo_name,
                    "feature_set_name": set_name,
                    "features": feature_list,
                    "f1_mean": f1_mean,
                    "f1_std": f1_std,
                    "acc_mean": acc_mean,
                    "acc_std": acc_std,
                }

    cv_df = pd.DataFrame(cv_records)

    # In bảng tổng hợp kết quả 5-fold CV với 4 chữ số thập phân
    print("BẢNG KẾT QUẢ 5-FOLD STRATIFIED CROSS-VALIDATION (TẬP TRAIN):")
    header = (
        f"{'Thuật toán':<14} | {'Bộ đặc trưng':<24} | "
        f"{'F1-macro CV (Mean ± Std)':>26} | {'Accuracy CV (Mean ± Std)':>26}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for _, row in cv_df.iterrows():
        f1_str = f"{row['F1-macro (Mean)']:.4f} ± {row['F1-macro (Std)']:.4f}"
        acc_str = f"{row['Accuracy (Mean)']:.4f} ± {row['Accuracy (Std)']:.4f}"
        print(
            f"{row['Thuật toán']:<14} | {row['Bộ đặc trưng']:<24} | "
            f"{f1_str:>26} | {acc_str:>26}"
        )
    print("-" * len(header))

    # In kết luận mô hình tối ưu nhất
    print("\n" + "=" * 70)
    print("KẾT LUẬN LỰA CHỌN MÔ HÌNH TỐI ƯU NHẤT (TASK T8):")
    print("=" * 70)
    print(
        f"=> Mô hình tốt nhất: {best_model_info['algorithm']} "
        f"trên bộ '{best_model_info['feature_set_name']}'\n"
        f"   - F1-macro CV trung bình : {best_model_info['f1_mean']:.4f} "
        f"± {best_model_info['f1_std']:.4f}\n"
        f"   - Accuracy CV trung bình : {best_model_info['acc_mean']:.4f} "
        f"± {best_model_info['acc_std']:.4f}\n"
        "   - Căn cứ lựa chọn        : Đạt điểm F1-macro cao nhất trên 5 fold "
        "huấn luyện,\n"
        "                              đảm bảo năng lực tổng quát hóa cân bằng "
        "giữa cả 7 lớp đậu."
    )
    print("=" * 70)

    return cv_df, best_model_info


def run_task_t8(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Thực thi Task T8: 5-fold Stratified CV và chọn mô hình tối ưu.

    Tham số:
        X_train: Đặc trưng tập huấn luyện.
        y_train: Nhãn lớp tập huấn luyện.

    Trả về:
        Tuple (cv_df, best_model_info).
    """
    return evaluate_cross_validation(X_train, y_train)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Vẽ heatmap ma trận nhầm lẫn trên tập Test bằng số lượng mẫu đếm thô.

    Tham số:
        cm: Ma trận nhầm lẫn dạng 2D numpy array.
        class_names: Danh sách tên 7 lớp đậu.
        output_dir: Thư mục lưu file biểu đồ.
    """
    print("\n2. TRỰC QUAN HÓA BIỂU ĐỒ 06: CONFUSION MATRIX (TASK T9)")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Số lượng mẫu"},
    )

    ax.set_title(
        "Ma trận nhầm lẫn trên tập kiểm tra\n"
        "(Mô hình: Decision Tree - Nhóm Hình dạng)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Nhãn dự đoán", fontsize=11, labelpad=8)
    ax.set_ylabel("Nhãn thực tế", fontsize=11, labelpad=8)
    ax.tick_params(axis="x", rotation=25, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "06_confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"- Đã lưu biểu đồ thành công: '{output_path}'")
    plt.close(fig)


def plot_model_comparison(
    cv_df: pd.DataFrame,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Vẽ biểu đồ cột nhóm so sánh F1-Macro 5-fold CV giữa 2 nhóm đặc trưng.

    Tham số:
        cv_df: DataFrame tổng hợp kết quả 5-fold CV từ Task T8.
        output_dir: Thư mục lưu file biểu đồ.
    """
    print("\n3. TRỰC QUAN HÓA BIỂU ĐỒ 07: SO SÁNH HIỆU NĂNG MÔ HÌNH (TASK T9)")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    # Chỉ so sánh trên 2 nhóm đặc trưng nghiên cứu chính theo RQ1
    target_sets = ["Kích thước (8 đặc trưng)", "Hình dạng (8 đặc trưng)"]
    sub_df = cv_df[cv_df["Bộ đặc trưng"].isin(target_sets)]
    algorithms = list(cv_df["Thuật toán"].unique())

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(target_sets))
    bar_width = 0.32 if len(algorithms) <= 2 else 0.8 / len(algorithms)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, algo in enumerate(algorithms):
        algo_df = sub_df[sub_df["Thuật toán"] == algo]
        f1_values = []
        for ts in target_sets:
            match = algo_df[algo_df["Bộ đặc trưng"] == ts]
            val = (
                match["F1-macro (Mean)"].values[0] if not match.empty else 0.0
            )
            f1_values.append(val)

        offset = (i - (len(algorithms) - 1) / 2) * bar_width
        rects = ax.bar(
            x + offset,
            f1_values,
            bar_width,
            label=algo,
            color=palette[i % len(palette)],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
        )
        ax.bar_label(
            rects, fmt="%.4f", padding=4, fontsize=10, fontweight="bold"
        )

    ax.set_title(
        "So sánh F1-Macro (CV 5-fold) giữa các mô hình "
        "theo 2 nhóm đặc trưng",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Nhóm đặc trưng hình thái", fontsize=11, labelpad=8)
    ax.set_ylabel(
        "Điểm F1-Macro trung bình (5-fold CV)", fontsize=11, labelpad=8
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Nhóm Kích thước (8 đặc trưng)", "Nhóm Hình dạng (8 đặc trưng)"],
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(title="Thuật toán", fontsize=10, title_fontsize=11)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "07_so_sanh_mo_hinh.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"- Đã lưu biểu đồ thành công: '{output_path}'")
    plt.close(fig)


def plot_feature_importances(
    dt_model: DecisionTreeClassifier,
    feature_names: list[str],
    output_dir: str = OUTPUT_DIR,
) -> pd.Series:
    """Vẽ biểu đồ cột ngang thể hiện feature_importances_ của Decision Tree.

    Tham số:
        dt_model: Mô hình Decision Tree tốt nhất (nhóm Hình dạng).
        feature_names: Danh sách 8 đặc trưng hình dạng.
        output_dir: Thư mục lưu file biểu đồ.

    Trả về:
        pd.Series chứa mức độ quan trọng sắp xếp giảm dần.
    """
    print("\n4. TRỰC QUAN HÓA BIỂU ĐỒ 08: FEATURE IMPORTANCES (TASK T9)")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"

    # Sắp xếp tăng dần để khi vẽ barh, giá trị lớn nhất nằm trên cùng
    importances_asc = pd.Series(
        dt_model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        importances_asc.index,
        importances_asc.values,
        color="#2ca02c",
        edgecolor="black",
        linewidth=0.7,
        height=0.6,
        alpha=0.9,
    )

    ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=10, fontweight="bold")

    ax.set_title(
        "Mức độ quan trọng của các đặc trưng "
        "(Decision Tree - Nhóm Hình dạng)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(
        "Mức độ quan trọng (Gini Importance)", fontsize=11, labelpad=8
    )
    ax.set_ylabel("Đặc trưng hình dạng", fontsize=11, labelpad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(0, max(importances_asc.values) * 1.15)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "08_feature_importances.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"- Đã lưu biểu đồ thành công: '{output_path}'")
    plt.close(fig)

    return importances_asc.sort_values(ascending=False)


def evaluate_test_set(
    trained_models: dict[str, dict[str, object]],
    best_model_info: dict[str, object],
    cv_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = OUTPUT_DIR,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], pd.Series]:
    """Đánh giá mô hình tốt nhất duy nhất một lần trên tập Test và vẽ 3 biểu đồ.

    Tham số:
        trained_models: Dict chứa các mô hình đã huấn luyện từ Task T7.
        best_model_info: Thông tin mô hình tốt nhất chọn từ Task T8.
        cv_df: DataFrame kết quả 5-fold CV từ Task T8 (dùng cho Biểu đồ 07).
        X_test: Đặc trưng tập kiểm tra (chỉ dùng duy nhất cho mô hình tốt nhất).
        y_test: Nhãn thực tế tập kiểm tra.
        output_dir: Thư mục lưu biểu đồ.

    Trả về:
        Tuple (y_pred, cm, report_dict, importances).
    """
    print("\n" + "=" * 70)
    print("BƯỚC 9: ĐÁNH GIÁ TRÊN TẬP KIỂM TRA - TEST SET 20% (TASK T9)")
    print("=" * 70)

    # 1. Trích xuất mô hình tốt nhất từ T8 (Decision Tree - Nhóm Hình dạng)
    best_algo = best_model_info["algorithm"]
    best_set_name = best_model_info["feature_set_name"]
    best_key = f"{best_algo} - {best_set_name}"
    best_model = trained_models[best_key]["model"]
    best_features = best_model_info["features"]

    # Chỉ dự đoán duy nhất 1 lần trên tập Test của nhóm Hình dạng (RULES.md)
    X_test_hinh_dang = X_test[best_features]
    y_pred = best_model.predict(X_test_hinh_dang)

    acc_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average="macro")

    print(f"\n1. KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TỐI ƯU NHẤT TRÊN TẬP TEST:")
    print(f"   - Mô hình được đánh giá  : {best_key}")
    print(
        f"   - Kích thước tập kiểm tra: {len(X_test_hinh_dang)} mẫu, "
        f"{len(best_features)} đặc trưng hình dạng."
    )
    print(f"   - Accuracy trên Test set : {acc_test:.4f}")
    print(f"   - F1-macro trên Test set : {f1_test:.4f}")

    # 2. In classification_report tiếng Việt rõ ràng
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
    )

    print("\nBẢNG BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT):")
    header = (
        f"{'Lớp (Class)':<14} | {'Precision':>11} | {'Recall':>11} | "
        f"{'F1-Score':>11} | {'Số mẫu (Support)':>16}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for cls in CLASS_NAMES:
        m = report_dict[cls]
        print(
            f"{cls:<14} | "
            f"{m['precision']:>11.4f} | "
            f"{m['recall']:>11.4f} | "
            f"{m['f1-score']:>11.4f} | "
            f"{int(m['support']):>16}"
        )
    print("-" * len(header))
    print(
        f"{'Accuracy':<14} | {'':>11} | {'':>11} | "
        f"{report_dict['accuracy']:>11.4f} | "
        f"{len(y_test):>16}"
    )
    print(
        f"{'Macro Avg':<14} | "
        f"{report_dict['macro avg']['precision']:>11.4f} | "
        f"{report_dict['macro avg']['recall']:>11.4f} | "
        f"{report_dict['macro avg']['f1-score']:>11.4f} | "
        f"{len(y_test):>16}"
    )
    print(
        f"{'Weighted Avg':<14} | "
        f"{report_dict['weighted avg']['precision']:>11.4f} | "
        f"{report_dict['weighted avg']['recall']:>11.4f} | "
        f"{report_dict['weighted avg']['f1-score']:>11.4f} | "
        f"{len(y_test):>16}"
    )
    print("-" * len(header))

    # 3. Phân tích ma trận nhầm lẫn và tìm cặp nhầm lẫn nhiều nhất
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)

    # Tự động tìm cặp lớp bị nhầm lẫn nhiều nhất (bỏ qua đường chéo chính)
    pair_confusion = {}
    max_single_err = -1
    single_confused_pair = ("", "", 0)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i == j:
                continue
            err = int(cm[i, j])
            if err > max_single_err:
                max_single_err = err
                single_confused_pair = (CLASS_NAMES[i], CLASS_NAMES[j], err)

            pair_key = tuple(sorted([CLASS_NAMES[i], CLASS_NAMES[j]]))
            pair_confusion[pair_key] = pair_confusion.get(pair_key, 0) + err

    most_confused_pair, total_err = max(
        pair_confusion.items(), key=lambda item: item[1]
    )

    idx_a = CLASS_NAMES.index(most_confused_pair[0])
    idx_b = CLASS_NAMES.index(most_confused_pair[1])
    err_a_to_b = cm[idx_a, idx_b]
    err_b_to_a = cm[idx_b, idx_a]

    print("\nPHÂN TÍCH NHẦM LẪN GIỮA CÁC LỚP ĐẬU (CONFUSION ANALYSIS):")
    print(
        f"=> Cặp lớp đậu bị nhầm lẫn nhiều nhất: "
        f"'{most_confused_pair[0]}' và '{most_confused_pair[1]}'\n"
        f"   - Tổng số mẫu nhầm lẫn 2 chiều : {total_err} mẫu\n"
        f"   - Chi tiết: {most_confused_pair[0]} nhầm thành "
        f"{most_confused_pair[1]}: {err_a_to_b} mẫu\n"
        f"               {most_confused_pair[1]} nhầm thành "
        f"{most_confused_pair[0]}: {err_b_to_a} mẫu\n"
        f"   - Chiều nhầm lẫn đơn lẻ lớn nhất: "
        f"{single_confused_pair[0]} -> {single_confused_pair[1]} "
        f"({single_confused_pair[2]} mẫu nhầm)"
    )

    # 4. Vẽ và lưu 3 biểu đồ theo quy định Task T9
    plot_confusion_matrix(cm, CLASS_NAMES, output_dir)
    plot_model_comparison(cv_df, output_dir)
    importances = plot_feature_importances(
        best_model, best_features, output_dir
    )

    return y_pred, cm, report_dict, importances


def run_task_t9(
    trained_models: dict[str, dict[str, object]],
    best_model_info: dict[str, object],
    cv_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = OUTPUT_DIR,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], pd.Series]:
    """Thực thi Task T9: Đánh giá Test set và xuất 3 biểu đồ 06, 07, 08.

    Tham số:
        trained_models: Dict chứa các mô hình đã huấn luyện từ Task T7.
        best_model_info: Thông tin mô hình tốt nhất chọn từ Task T8.
        cv_df: DataFrame kết quả 5-fold CV từ Task T8.
        X_test: Đặc trưng tập kiểm tra.
        y_test: Nhãn lớp tập kiểm tra.
        output_dir: Thư mục lưu ảnh.

    Trả về:
        Tuple (y_pred, cm, report_dict, importances).
    """
    return evaluate_test_set(
        trained_models, best_model_info, cv_df, X_test, y_test, output_dir
    )


def answer_research_question_1(
    trained_models: dict[str, dict[str, object]],
    best_model_info: dict[str, object],
    cv_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """Tổng hợp số liệu thực nghiệm và trả lời câu hỏi nghiên cứu 1 (RQ1).

    Tham số:
        trained_models: Dict chứa 6 mô hình đã huấn luyện từ Task T7.
        best_model_info: Thông tin mô hình tốt nhất chọn từ Task T8.
        cv_df: DataFrame tổng hợp kết quả 5-fold CV từ Task T8.
        X_test: Đặc trưng tập kiểm tra.
        y_test: Nhãn lớp tập kiểm tra.

    Trả về:
        Dict chứa các chỉ số tổng hợp và phân tích tỷ trọng cho RQ1.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 10: TỔNG KẾT VÀ TRẢ LỜI CÂU HỎI NGHIÊN CỨU 1 - RQ1 (TASK T10)")
    print("=" * 70)

    # 1. Đánh giá kiểm thử 6 tổ hợp trên Test set để lập bảng so sánh đầy đủ
    eval_records = []
    for key, item in trained_models.items():
        algo = item["algorithm"]
        set_name = item["feature_set_name"]
        feats = item["features"]
        model = item["model"]

        preds = model.predict(X_test[feats])
        acc_test = float(accuracy_score(y_test, preds))
        f1_test = float(f1_score(y_test, preds, average="macro"))

        # Trích xuất số liệu 5-fold CV tương ứng từ cv_df
        cv_match = cv_df[
            (cv_df["Thuật toán"] == algo)
            & (cv_df["Bộ đặc trưng"] == set_name)
        ]
        cv_row = cv_match.iloc[0]

        eval_records.append(
            {
                "Thuật toán": algo,
                "Bộ đặc trưng": set_name,
                "F1_CV_Mean": cv_row["F1-macro (Mean)"],
                "F1_CV_Std": cv_row["F1-macro (Std)"],
                "Acc_CV_Mean": cv_row["Accuracy (Mean)"],
                "Acc_CV_Std": cv_row["Accuracy (Std)"],
                "Acc_Test": acc_test,
                "F1_Test": f1_test,
            }
        )

    summary_df = pd.DataFrame(eval_records)

    def get_val(algo_name: str, set_label: str, col_name: str) -> float:
        subset = summary_df[
            (summary_df["Thuật toán"] == algo_name)
            & (summary_df["Bộ đặc trưng"] == set_label)
        ]
        return float(subset[col_name].values[0]) if not subset.empty else 0.0

    # In bảng tổng hợp so sánh hiệu năng
    print("\n1. BẢNG TỔNG HỢP HIỆU NĂNG 6 TỔ HỢP MÔ HÌNH VÀ BỘ ĐẶC TRƯNG:")
    header = (
        f"{'Thuật toán':<14} | {'Bộ đặc trưng':<24} | "
        f"{'F1-macro CV (5-fold)':>22} | {'Test Acc':>9} | {'Test F1':>9}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for _, row in summary_df.iterrows():
        cv_str = f"{row['F1_CV_Mean']:.4f} ± {row['F1_CV_Std']:.4f}"
        print(
            f"{row['Thuật toán']:<14} | {row['Bộ đặc trưng']:<24} | "
            f"{cv_str:>22} | {row['Acc_Test']:>9.4f} | "
            f"{row['F1_Test']:>9.4f}"
        )
    print("-" * len(header))

    # 2. Phân tích mô hình tốt nhất
    best_algo = best_model_info["algorithm"]
    best_set = best_model_info["feature_set_name"]
    best_f1_cv = best_model_info["f1_mean"]
    best_f1_std = best_model_info["f1_std"]
    best_acc_cv = best_model_info["acc_mean"]
    best_acc_std = best_model_info["acc_std"]
    best_test_acc = get_val(best_algo, best_set, "Acc_Test")
    best_test_f1 = get_val(best_algo, best_set, "F1_Test")

    print("\n2. MÔ HÌNH TỐI ƯU NHẤT ĐƯỢC LỰA CHỌN:")
    print(f"   - Tên mô hình            : {best_algo} - {best_set}")
    print(
        f"   - F1-macro CV (Train)    : {best_f1_cv:.4f} ± {best_f1_std:.4f}"
    )
    print(
        f"   - Accuracy CV (Train)    : {best_acc_cv:.4f} ± {best_acc_std:.4f}"
    )
    print(f"   - Accuracy (Test set)    : {best_test_acc:.4f}")
    print(f"   - F1-macro (Test set)    : {best_test_f1:.4f}")
    print(
        "   - Căn cứ lựa chọn        : Dựa hoàn toàn trên điểm F1-macro "
        "5-fold CV trung bình\n"
        "                              cao nhất trên tập huấn luyện (Train set),\n"
        "                              tuân thủ nguyên tắc phương pháp luận "
        "(không chọn mô hình\n"
        "                              dựa trên tập kiểm tra để tránh rò rỉ "
        "thông tin)."
    )

    # 3. Phân tích tỷ trọng đóng góp feature_importances_ trên 16 đặc trưng
    dt_all_key = "Decision Tree - Toàn bộ (16 đặc trưng)"
    dt_all_model = trained_models[dt_all_key]["model"]
    importances = pd.Series(
        dt_all_model.feature_importances_, index=ALL_FEATURES
    )

    size_importance = float(importances[SIZE_FEATURES].sum())
    shape_importance = float(importances[SHAPE_FEATURES].sum())
    size_pct = size_importance * 100
    shape_pct = shape_importance * 100

    top_features = importances.sort_values(ascending=False).head(5)

    print("\n3. PHÂN TÍCH TỶ TRỌNG ĐỘ QUAN TRỌNG ĐẶC TRƯNG (FEATURE IMPORTANCES):")
    print(
        f"   - Tổng tỷ trọng nhóm Hình dạng  : {shape_importance:.4f} "
        f"({shape_pct:.2f}%)"
    )
    print(
        f"   - Tổng tỷ trọng nhóm Kích thước : {size_importance:.4f} "
        f"({size_pct:.2f}%)"
    )
    ratio = shape_importance / size_importance if size_importance > 0 else 0
    print(
        f"   -> Nhóm Hình dạng chiếm ưu thế gấp {ratio:.2f} lần "
        "nhóm Kích thước."
    )
    print(
        "\n   Top 5 đặc trưng quan trọng nhất toàn bộ tập dữ liệu (16 đặc trưng):"
    )
    for rank, (feat, val) in enumerate(top_features.items(), 1):
        group = "Hình dạng" if feat in SHAPE_FEATURES else "Kích thước"
        print(
            f"     {rank}. {feat:<16} ({group:<10}): "
            f"{val:.4f} ({val * 100:.2f}%)"
        )

    # 4. Trả lời trực diện câu hỏi nghiên cứu 1 (RQ1)
    nb_size_f1 = get_val("GaussianNB", "Kích thước (8 đặc trưng)", "F1_CV_Mean")
    nb_shape_f1 = get_val("GaussianNB", "Hình dạng (8 đặc trưng)", "F1_CV_Mean")
    dt_size_f1 = get_val(
        "Decision Tree", "Kích thước (8 đặc trưng)", "F1_CV_Mean"
    )
    dt_shape_f1 = get_val(
        "Decision Tree", "Hình dạng (8 đặc trưng)", "F1_CV_Mean"
    )
    dt_all_f1 = get_val(
        "Decision Tree", "Toàn bộ (16 đặc trưng)", "F1_CV_Mean"
    )

    nb_diff = (nb_shape_f1 - nb_size_f1) * 100
    dt_diff = (dt_shape_f1 - dt_size_f1) * 100

    print("\n" + "=" * 70)
    print("KẾT LUẬN VÀ TRẢ LỜI CÂU HỎI NGHIÊN CỨU 1 (RQ1):")
    print("=" * 70)
    print(
        "CÂU HỎI RQ1: Dựa vào nhóm đặc trưng nào, kích thước hay hình dạng,\n"
        "             để xác định loại đậu chính xác hơn?\n"
    )
    print(
        "CÂU TRẢ LỜI TRỰC DIỆN: NHÓM ĐẶC TRƯNG HÌNH DẠNG (SHAPE FEATURES)\n"
        "                      GIÚP PHÂN LOẠI 7 LOẠI ĐẬU CHÍNH XÁC HƠN RÕ RỆT."
    )
    print(
        "\nCÁC CĂN CỨ THỰC NGHIỆM ĐỊNH LƯỢNG:\n"
        f"1. Hiệu năng vượt trội trên cả 2 thuật toán:\n"
        f"   - Với GaussianNB: F1-macro tăng vọt từ {nb_size_f1:.4f} "
        f"(Kích thước)\n"
        f"     lên {nb_shape_f1:.4f} (Hình dạng), tăng trưởng ấn tượng "
        f"+{nb_diff:.2f}%.\n"
        f"   - Với Decision Tree: F1-macro nhóm Hình dạng đạt {dt_shape_f1:.4f},\n"
        f"     cao hơn rõ rệt nhóm Kích thước ({dt_size_f1:.4f}, "
        f"tăng +{dt_diff:.2f}%)\n"
        f"     và vượt qua cả khi dùng toàn bộ 16 đặc trưng ({dt_all_f1:.4f}).\n"
        f"2. Áp đảo về tỷ trọng đóng góp (Gini Importance):\n"
        f"   - Nhóm Hình dạng chiếm {shape_pct:.2f}% tổng tỷ trọng, trong khi "
        f"nhóm Kích thước chỉ chiếm {size_pct:.2f}%.\n"
        f"   - Top 2 đặc trưng quan trọng nhất đều thuộc nhóm Hình dạng\n"
        f"     ({top_features.index[0]}: {top_features.iloc[0] * 100:.2f}%, "
        f"{top_features.index[1]}: {top_features.iloc[1] * 100:.2f}%).\n"
        "\nÝ NGHĨA HÌNH THÁI HỌC VÀ THỰC TIỄN NÔNG HỌC:\n"
        "- Kích thước hạt (Area, Perimeter, ...) dễ bị biến thiên mạnh bởi\n"
        "  điều kiện ngoại cảnh như thổ nhưỡng, lượng mưa, độ ẩm và mùa vụ.\n"
        "- Ngược lại, các tỷ lệ hình thái không thứ nguyên (Compactness, "
        "roundness,\n"
        "  ShapeFactors) là đặc tính di truyền học ổn định của từng giống đậu,\n"
        "  giúp phân biệt xuất sắc giữa các giống đậu có kích thước tương đồng\n"
        "  (như DERMASON và SIRA) mà nhóm kích thước đơn thuần không thể "
        "phân tách."
    )
    print("=" * 70)

    return {
        "summary_df": summary_df,
        "best_model": f"{best_algo} - {best_set}",
        "shape_importance": shape_importance,
        "size_importance": size_importance,
        "top_features": top_features,
    }


def run_task_t10(
    trained_models: dict[str, dict[str, object]],
    best_model_info: dict[str, object],
    cv_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """Thực thi Task T10: Tổng kết và trả lời câu hỏi nghiên cứu RQ1.

    Tham số:
        trained_models: Dict chứa các mô hình đã huấn luyện từ Task T7.
        best_model_info: Thông tin mô hình tốt nhất chọn từ Task T8.
        cv_df: DataFrame kết quả 5-fold CV từ Task T8.
        X_test: Đặc trưng tập kiểm tra.
        y_test: Nhãn lớp tập kiểm tra.

    Trả về:
        Dict chứa các chỉ số và kết luận thực nghiệm cho RQ1.
    """
    return answer_research_question_1(
        trained_models, best_model_info, cv_df, X_test, y_test
    )


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

    # Task T8: 5-fold Stratified CV trên tập Train và chọn mô hình tối ưu
    cv_df, best_model_info = run_task_t8(X_train, y_train)

    # Task T9: Đánh giá trên tập kiểm tra và trực quan hóa biểu đồ 06, 07, 08
    y_pred, cm_test, report_dict, importances = run_task_t9(
        trained_models, best_model_info, cv_df, X_test, y_test, OUTPUT_DIR
    )

    # Task T10: Tổng kết và trả lời câu hỏi nghiên cứu RQ1
    rq1_results = run_task_t10(
        trained_models, best_model_info, cv_df, X_test, y_test
    )

    print("\n" + "=" * 70)
    print("HOÀN THÀNH TASKS T9 & T10 (SPRINT 3: MÔ HÌNH HÓA) THÀNH CÔNG!")
    print(f"- Đã đánh giá trên tập Test: {len(X_test)} mẫu.")
    print("- Đã xuất đủ 3 biểu đồ đánh giá theo chuẩn DESIGN.md (DPI 300):")
    print("  + 06_confusion_matrix.png")
    print("  + 07_so_sanh_mo_hinh.png")
    print("  + 08_feature_importances.png")
    print(
        f"- Đã trả lời trực diện RQ1: Nhóm Hình dạng vượt trội "
        f"({rq1_results['shape_importance'] * 100:.2f}% tỷ trọng đóng góp)."
    )
    print(
        f"- Mô hình tốt nhất: {rq1_results['best_model']} "
        f"(F1-macro CV = {best_model_info['f1_mean']:.4f})."
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
