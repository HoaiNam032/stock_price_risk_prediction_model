import pandas as pd
import numpy as np
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


# =========================
# 1. Đọc & tính toán số liệu
# =========================

def build_margin_report_df(
    excel_path: str,
    equity_vnd_million: float = 2_077_000.0,
    special_room_path: str | None = None,
) -> pd.DataFrame:
    """
    Đọc file raw (gồm các sheet:
      - Margin List
      - 230007
      - Vol list
      - Matched result
    ) và trả về DataFrame đã tính toán cho report.

    Nếu có special_room_path (file '1 số room đặc biệt.xlsx') thì
    sẽ tính thêm theo từng mã:
      - Total Used room (triệu VND) theo rule group (Stock, Maximum loan price)
      - Giá chặn riêng (Maximum loan price min theo mã)
    """

    # 1. Đọc dữ liệu
    margin_df = pd.read_excel(excel_path, sheet_name="Margin List")
    used_df   = pd.read_excel(excel_path, sheet_name="230007")
    vol_df    = pd.read_excel(excel_path, sheet_name="Vol list")
    price_df  = pd.read_excel(excel_path, sheet_name="Matched result")

    # 2. Chuẩn hoá tên cột để merge
    margin_df = margin_df.rename(columns={"Mã CK": "code"})
    used_df   = used_df.rename(columns={"Mã chứng khoán": "code"})
    vol_df    = vol_df.rename(columns={"Stock": "code"})
    price_df  = price_df.rename(columns={"Mã": "code"})

    # 3. Lấy các cột cần thiết
    used_df = used_df[["code", "Room hệ thống đã sử dụng", "Room đặc biệt đã sử dụng"]]
    margin_df = margin_df[
        ["code", "Tỷ lệ vay KQ (%)", "Tỷ lệ vay TC  (%)", "Giá vay/Giá TSĐB tối đa (VND)"]
    ]
    vol_df   = vol_df[["code", "Vol-listed"]]
    price_df = price_df[["code", "Đóng\n cửa"]]

    # 4. Merge lại thành 1 bảng
    df = (
        used_df
        .merge(margin_df, on="code", how="left")
        .merge(vol_df,   on="code", how="left")
        .merge(price_df, on="code", how="left")
    )

    # 5. Tính toán các chỉ tiêu cơ bản

    # (3) Số lượng CK niêm yết: nghìn cổ phiếu
    df["Số lượng chứng khoán niêm yết (3)"] = df["Vol-listed"] / 1000.0

    # Tổng room đã dùng (cổ phiếu)
    total_used_shares = (
        df["Room hệ thống đã sử dụng"].fillna(0)
        + df["Room đặc biệt đã sử dụng"].fillna(0)
    )

    # Giới hạn 5% khối lượng niêm yết (cổ phiếu)
    limit_5pct_shares = 0.05 * df["Vol-listed"].fillna(0)

    # Số lượng cổ phiếu thực sự được phép cho vay (đã chặn 5% vol)
    allowed_lend_shares = np.minimum(total_used_shares, limit_5pct_shares)

    # (2) Số lượng CK cho vay của CTCK (nghìn cổ phiếu)
    df["Số lượng chứng khoán cho vay (2)"] = allowed_lend_shares / 1000.0

    # Reference Price = Giá đóng cửa * 1000 (file raw để đơn vị nghìn)
    df["Reference Price"] = df["Đóng\n cửa"] * 1000.0

    # Tỷ lệ vay & giá tối đa
    df["MR Loan ratio"] = df["Tỷ lệ vay KQ (%)"]
    df["DP Loan ratio"] = df["Tỷ lệ vay TC  (%)"]
    df["Max Price"]     = df["Giá vay/Giá TSĐB tối đa (VND)"]

    # Used GR / Used SR (để hiển thị)
    df["Used GR"] = df["Room hệ thống đã sử dụng"]
    df["Used SR"] = df["Room đặc biệt đã sử dụng"]

    # (4) Vốn CSH CTCK (triệu đồng)
    df["Vốn chủ sở hữu của CTCK (4)"] = equity_vnd_million

    # ===== Công thức DƯ NỢ (1) =====
    # base_vnd = allowed_lend_shares * min(Ref, MaxPrice) * max(MR, DP)/100
    # Dư nợ (triệu) = min( max(base_vnd, 0) / 1e6 , 10% * VCSH )

    million_factor = 1_000_000.0  # đơn vị Dư nợ là triệu đồng

    # min(reference, max price)
    price_cap = df[["Reference Price", "Max Price"]].min(axis=1)

    # max(MR loan ratio, DP loan ratio)
    ratio_cap = df[["MR Loan ratio", "DP Loan ratio"]].max(axis=1)

    base_vnd = allowed_lend_shares * price_cap * ratio_cap / 100.0

    base_debt_million   = base_vnd.clip(lower=0) / million_factor
    max_debt_by_equity  = 0.10 * df["Vốn chủ sở hữu của CTCK (4)"]

    df["Dư nợ cho vay GDKQ (1)"] = np.minimum(base_debt_million, max_debt_by_equity)

    # Tỷ lệ dư nợ/VCSH
    df["Tỷ lệ dư nợ/VCSH (1)/(4)"] = (
        df["Dư nợ cho vay GDKQ (1)"] / df["Vốn chủ sở hữu của CTCK (4)"]
    )

    # Tỷ lệ CK cho vay/CKNY
    df["Tỷ lệ CK cho vay/CKNY (2)/(3)"] = (
        df["Số lượng chứng khoán cho vay (2)"] / df["Số lượng chứng khoán niêm yết (3)"]
    )

    # Chuẩn bị cột Total Used room / Giá chặn riêng (mặc định None)
    df["Total Used room"] = np.nan   # sẽ là TRIỆU VND
    df["Giá chặn riêng"]  = np.nan   # Maximum loan price min theo mã

    # ===== GHÉP THÊM ROOM ĐẶC BIỆT TỪ FILE '1 số room đặc biệt.xlsx' (NẾU CÓ) =====
    if special_room_path is not None:
        room_df = pd.read_excel(special_room_path)

        # Chuẩn hoá tên mã
        if "Stock " in room_df.columns:
            room_df = room_df.rename(columns={"Stock ": "code"})
        elif "Stock" in room_df.columns:
            room_df = room_df.rename(columns={"Stock": "code"})

        # Ép numeric: '-' hoặc rỗng -> NaN -> 0
        num_cols = [
            "Used quantity today",
            "Maximum loan price",
            " MR Approved Ratio (%)",
            " DP Approved Ratio (%)",
        ]
        for col in num_cols:
            if col in room_df.columns:
                room_df[col] = pd.to_numeric(room_df[col], errors="coerce").fillna(0)

        # Tỉ lệ theo TỪNG DÒNG
        room_df["ratio_row"] = room_df[
            [" MR Approved Ratio (%)", " DP Approved Ratio (%)"]
        ].min(axis=1)

        # --- QUAN TRỌNG: group theo (code, Maximum loan price) ---
        # Mục tiêu: dùng chung maximum loan price thì cộng Used quantity today lại
        # rồi nhân 1 lần với loan price & min(MR, DP)
        grouped = (
            room_df
            .groupby(["code", "Maximum loan price"], as_index=False)
            .agg(
                Used_qty_sum=("Used quantity today", "sum"),
                MR_min=(" MR Approved Ratio (%)", "min"),
                DP_min=(" DP Approved Ratio (%)", "min"),
            )
        )

        grouped["ratio_group"] = grouped[["MR_min", "DP_min"]].min(axis=1)

        # Dư nợ từng GROUP (VND)
        grouped["GroupLoanVND"] = (
            grouped["Used_qty_sum"]
            * grouped["Maximum loan price"]
            * grouped["ratio_group"] / 100.0
        )

        # Tổng theo MÃ:
        #   - Total Used room (triệu VND)
        #   - Giá chặn riêng = min(Maximum loan price) theo mã
        agg_room = (
            grouped.groupby("code", as_index=False)
            .agg(
                Total_Used_room_million=("GroupLoanVND", lambda x: x.sum() / 1_000_000.0),
                Gia_chan_rieng=("Maximum loan price", "min"),
            )
        )

        agg_room = agg_room.rename(
            columns={
                "Total_Used_room_million": "Total Used room",
                "Gia_chan_rieng": "Giá chặn riêng",
            }
        )

        # Xoá 2 cột rỗng mặc định rồi merge cột chuẩn
        df = df.drop(columns=["Total Used room", "Giá chặn riêng"])
        df = df.merge(agg_room, on="code", how="left")

    # 6. Sắp xếp cột theo đúng logic báo cáo
    ordered_cols = [
        "code",
        "Dư nợ cho vay GDKQ (1)",
        "Số lượng chứng khoán cho vay (2)",
        "Số lượng chứng khoán niêm yết (3)",
        "Vốn chủ sở hữu của CTCK (4)",
        "Tỷ lệ dư nợ/VCSH (1)/(4)",
        "Tỷ lệ CK cho vay/CKNY (2)/(3)",
        "Used GR",
        "Used SR",
        "Reference Price",
        "MR Loan ratio",
        "DP Loan ratio",
        "Max Price",
        "Total Used room",
        "Giá chặn riêng",
    ]

    df_out = df[ordered_cols].copy()

    # Sort theo mã CK
    df_out = df_out.sort_values("code").reset_index(drop=True)

    # Thêm STT
    df_out.insert(0, "STT", range(1, len(df_out) + 1))

    return df_out


def num_or_dash(x):
    """Trả về '-' nếu x = 0 hoặc NaN, ngược lại trả về float(x)."""
    if pd.isna(x):
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return x
    return "-" if abs(v) < 1e-9 else v


# =========================
# 2A. File 1: Copy raw + thêm sheet Dư nợ theo used room_PY
# =========================

def write_margin_report_sheet(
    excel_input_path: str,
    excel_output_path: str,
    sheet_name: str = "Dư nợ theo used room_PY",
    equity_vnd_million: float = 2_077_000.0,
    special_room_path: str | None = None,
):
    """
    - Đọc file input (raw)
    - Tính DataFrame báo cáo margin
    - Mở workbook gốc và thêm 1 sheet report mới (giống kiểu 'Dư nợ theo used room')
    - Lưu ra excel_output_path
    """

    # 1. Tính toán DataFrame
    df = build_margin_report_df(
        excel_input_path,
        equity_vnd_million=equity_vnd_million,
        special_room_path=special_room_path,
    )

    # Lấy danh sách mã có trong file room đặc biệt (để tô vàng)
    special_codes = set()
    if special_room_path is not None:
        room_code_df = pd.read_excel(special_room_path)

        # Chuẩn hoá tên cột mã
        if "Stock " in room_code_df.columns:
            room_code_df = room_code_df.rename(columns={"Stock ": "code"})
        elif "Stock" in room_code_df.columns:
            room_code_df = room_code_df.rename(columns={"Stock": "code"})

        if "code" in room_code_df.columns:
            special_codes = set(
                str(x).strip()
                for x in room_code_df["code"].dropna().unique()
            )


    # 2. Mở workbook gốc
    wb = load_workbook(excel_input_path)

    # Nếu sheet này đã tồn tại thì xoá trước (tránh trùng)
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]

    ws = wb.create_sheet(title=sheet_name)

    # 3. Header giống logic 'Dư nợ theo used room'
    headers = [
        "STT",
        "Mã chứng khoán",
        "Dư nợ cho vay GDKQ \n(1)",
        "Số lượng chứng khoán cho vay của CTCK \n(2)",
        "Số lượng chứng khoán niêm yết của TCNY\n(3)",
        "Vốn chủ sở hữu của CTCK \n(4)",
        "Tỷ lệ dư nợ/VCSH \n(1)/(4)",
        "Tỷ lệ CK cho vay/CKNY\n(2)/(3)",
        "Used GR",
        "Used SR",
        "Reference Price",
        "MR Loan ratio",
        "DP Loan ratio",
        "Max Price",
        "Control",
        "Total Used room",
        "Giá chặn riêng",
    ]
    n_cols = len(headers)

    # 4. Tiêu đề + dòng đơn vị
    ws.cell(row=1, column=1, value="TÌNH HÌNH GIAO DỊCH KÝ QUỸ")
    ws.cell(row=2, column=1, value="Đơn vị : nghìn cổ phiếu,triệu đồng")

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=8)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=8)

    # 5. Header ở hàng 3
    for col_idx, header in enumerate(headers, start=1):
        ws.cell(row=3, column=col_idx, value=header)

    # 6. Ghi dữ liệu (hàng 4 trở đi)
    for row_idx, row in df.iterrows():
        excel_row = row_idx + 4

        ws.cell(row=excel_row, column=1, value=int(row["STT"]))
        ws.cell(row=excel_row, column=2, value=row["code"])

        ws.cell(row=excel_row, column=3, value=num_or_dash(row["Dư nợ cho vay GDKQ (1)"]))
        ws.cell(row=excel_row, column=4, value=num_or_dash(row["Số lượng chứng khoán cho vay (2)"]))
        ws.cell(row=excel_row, column=5, value=num_or_dash(row["Số lượng chứng khoán niêm yết (3)"]))

        # VCSH
        ws.cell(row=excel_row, column=6, value=num_or_dash(row["Vốn chủ sở hữu của CTCK (4)"]))

        # 2 cột tỷ lệ GIỮ LÀ SỐ (để Excel format 0.00%)
        ws.cell(
            row=excel_row,
            column=7,
            value=float(row["Tỷ lệ dư nợ/VCSH (1)/(4)"]) if pd.notna(row["Tỷ lệ dư nợ/VCSH (1)/(4)"]) else None,
        )
        ws.cell(
            row=excel_row,
            column=8,
            value=float(row["Tỷ lệ CK cho vay/CKNY (2)/(3)"]) if pd.notna(row["Tỷ lệ CK cho vay/CKNY (2)/(3)"]) else None,
        )

        ws.cell(row=excel_row, column=9,  value=num_or_dash(row["Used GR"]))
        ws.cell(row=excel_row, column=10, value=num_or_dash(row["Used SR"]))
        ws.cell(row=excel_row, column=11, value=num_or_dash(row["Reference Price"]))
        ws.cell(row=excel_row, column=12, value=num_or_dash(row["MR Loan ratio"]))
        ws.cell(row=excel_row, column=13, value=num_or_dash(row["DP Loan ratio"]))
        ws.cell(row=excel_row, column=14, value=num_or_dash(row["Max Price"]))

        # 15: Control (chưa có rule → để trống)
        # 16: Total Used room (triệu VND) từ file 1 số room đặc biệt nếu có
        ws.cell(
            row=excel_row,
            column=16,
            value=num_or_dash(row.get("Total Used room", np.nan)),
        )
        # 17: Giá chặn riêng
        ws.cell(
            row=excel_row,
            column=17,
            value=num_or_dash(row.get("Giá chặn riêng", np.nan)),
        )

    # 6b. Hàng tổng
    total_row = 4 + len(df)
    total_debt = df["Dư nợ cho vay GDKQ (1)"].sum()
    total_qty  = df["Số lượng chứng khoán cho vay (2)"].sum()

    ws.cell(row=total_row, column=3, value=num_or_dash(total_debt))
    ws.cell(row=total_row, column=4, value=num_or_dash(total_qty))

    # =========================
    # 7. Format
    # =========================

    # Style cơ bản
    title_font  = Font(bold=True, size=13)
    unit_font   = Font(italic=True)
    header_font = Font(bold=True, color="000000")
    header_fill = PatternFill(start_color="FFC867", end_color="FFC867", fill_type="solid")
    thin        = Side(border_style="thin", color="000000")
    border      = Border(left=thin, right=thin, top=thin, bottom=thin)
    center      = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Tiêu đề
    ws["A1"].font      = title_font
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws["A2"].font      = unit_font
    ws["A2"].alignment = Alignment(horizontal="right", vertical="center")

    # Header row style
    for col_idx in range(1, n_cols + 1):
        cell = ws.cell(row=3, column=col_idx)
        cell.font      = header_font
        cell.alignment = center

    # Tô màu header cho các cột 9–14 (Used GR → Max Price)
    for col in range(9, 15):
        ws.cell(row=3, column=col).fill = header_fill

    # KẺ BẢNG: cột 1–14, header + data (không kẻ dòng tổng)
    first_header_row = 3
    first_data_row   = 4
    last_data_row    = 3 + len(df)

    # Border header
    for row in ws.iter_rows(
        min_row=first_header_row,
        max_row=first_header_row,
        min_col=1,
        max_col=14,
    ):
        for cell in row:
            cell.border = border

    # Border data
    for row in ws.iter_rows(
        min_row=first_data_row,
        max_row=last_data_row,
        min_col=1,
        max_col=14,
    ):
        for cell in row:
            cell.border = border

    # In đậm hàng tổng (cột 3 & 4), không kẻ border thêm
    bold_font = Font(bold=True)
    for col_idx in (3, 4):
        ws.cell(row=total_row, column=col_idx).font = bold_font

    # Tô vàng nhạt cho giá trị cột 9–14 (Used GR → Max Price)
    value_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    for row in ws.iter_rows(
        min_row=first_data_row,
        max_row=last_data_row,
        min_col=9,
        max_col=14,
    ):
        for cell in row:
            cell.fill = value_fill

    # Tô vàng cho các mã có trong file "1 số room đặc biệt":
    # từ cột 2 (Mã chứng khoán) đến cột 17 (Giá chặn riêng)
    highlight_fill = PatternFill(start_color="FFFF66", end_color="FFFF66", fill_type="solid")

    if special_codes:
        for excel_row in range(first_data_row, last_data_row + 1):
            code_val = ws.cell(row=excel_row, column=2).value  # cột "Mã chứng khoán"
            if code_val is None:
                continue

            code_str = str(code_val).strip()
            if code_str in special_codes:
                for col_idx in range(2, 18):  # từ cột B (2) đến cột Q (17)
                    ws.cell(row=excel_row, column=col_idx).fill = highlight_fill


    # Set width cột
    width_map = {
        1: 5,  2: 12, 3: 12, 4: 12,
        5: 15, 6: 12, 7: 12, 8: 12,
        9: 12, 10: 12, 11: 12, 12: 12,
        13: 12, 14: 12, 15: 12, 16: 12, 17: 12,
    }
    for col_idx, width in width_map.items():
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = width

    # Number format cho các cột số (bao gồm hàng tổng)
    for row in ws.iter_rows(min_row=4, max_row=total_row, min_col=1, max_col=n_cols):
        # Dư nợ (1) – cột 3
        row[2].number_format = "#,##0"
        # Số lượng CK cho vay (2), niêm yết (3) – cột 4,5
        row[3].number_format = "#,##0"
        row[4].number_format = "#,##0"
        # VCSH – cột 6
        row[5].number_format = "#,##0"
        # Tỷ lệ – cột 7,8 (%)
        row[6].number_format = "0.00%"
        row[7].number_format = "0.00%"
        # Used GR / Used SR – cột 9,10
        row[8].number_format = "#,##0"
        row[9].number_format = "#,##0"
        # Reference Price / Max Price – cột 11,14
        row[10].number_format = "#,##0"
        row[13].number_format = "#,##0"
        # MR/DP Loan ratio – cột 12,13
        row[11].number_format = "0"
        row[12].number_format = "0"
        # Total Used room – cột 16
        row[15].number_format = "#,##0"
        # Giá chặn riêng – cột 17
        row[16].number_format = "#,##0"

    # Canh phải cột số (3–17)
    right_align = Alignment(horizontal="right", vertical="center")
    for row in ws.iter_rows(
        min_row=4,
        max_row=total_row,
        min_col=3,
        max_col=17,
    ):
        for cell in row:
            cell.alignment = right_align

    # Freeze panes: cố định cột A–H và 3 hàng đầu
    ws.freeze_panes = "I4"

    # 8. Lưu file
    wb.save(excel_output_path)
    print(f"Đã tạo báo cáo '{sheet_name}' vào file: {excel_output_path}")


# =========================
# 2B. File 2: Tạo file 'Báo cáo ngày ..._PY.xlsx' mới từ raw
# =========================

def write_margin_report_file(
    raw_excel_path: str,
    output_report_path: str,
    sheet_name: str = "1.f_thgdkq_06692",
    equity_vnd_million: float = 2_077_000.0,
    special_room_path: str | None = None,
):
    """
    - Đọc file RAW
    - Tính DataFrame báo cáo margin
    - Tạo workbook mới với 1 sheet, format giống báo cáo ngày:
        + 8 cột: STT → Tỷ lệ CK cho vay/CKNY
        + Header nền xanh lá
        + Dòng 'Tổng cộng' ở cuối
    """

    # 1. Tính toán DataFrame đầy đủ
    df_full = build_margin_report_df(
        raw_excel_path,
        equity_vnd_million=equity_vnd_million,
        special_room_path=special_room_path,
    )

    # 2. Chỉ lấy 8 cột cần cho báo cáo ngày
    df = df_full[
        [
            "STT",
            "code",
            "Dư nợ cho vay GDKQ (1)",
            "Số lượng chứng khoán cho vay (2)",
            "Số lượng chứng khoán niêm yết (3)",
            "Vốn chủ sở hữu của CTCK (4)",
            "Tỷ lệ dư nợ/VCSH (1)/(4)",
            "Tỷ lệ CK cho vay/CKNY (2)/(3)",
        ]
    ].copy()

    # 3. Tạo workbook mới
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    # 4. Header & cấu trúc
    headers = [
        "STT",
        "Mã chứng khoán",
        "Dư nợ cho vay GDKQ (1)",
        "Số lượng chứng khoán cho vay của CTCK (2)",
        "Số lượng chứng khoán niêm yết của TCNY (3)",
        "Vốn chủ sở hữu của CTCK (4)",
        "Tỷ lệ dư nợ/VCSH (1)/(4)",
        "Tỷ lệ CK cho vay/CKNY (2)/(3)",
    ]
    n_cols = len(headers)

    # 5. Tiêu đề + dòng đơn vị
    ws.cell(row=1, column=1, value="TÌNH HÌNH GIAO DỊCH KÝ QUỸ")
    ws.cell(row=2, column=1, value="Đơn vị: nghìn cổ phiếu, triệu đồng")

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=n_cols)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=n_cols)

    # 6. Ghi dòng header (hàng 3)
    for col_idx, header in enumerate(headers, start=1):
        ws.cell(row=3, column=col_idx, value=header)

    # 7. Ghi dữ liệu (hàng 4 trở đi)
    first_data_row = 4
    for row_idx, row in df.iterrows():
        excel_row = first_data_row + row_idx

        ws.cell(row=excel_row, column=1, value=int(row["STT"]))
        ws.cell(row=excel_row, column=2, value=row["code"])

        ws.cell(row=excel_row, column=3, value=num_or_dash(row["Dư nợ cho vay GDKQ (1)"]))
        ws.cell(row=excel_row, column=4, value=num_or_dash(row["Số lượng chứng khoán cho vay (2)"]))
        ws.cell(row=excel_row, column=5, value=num_or_dash(row["Số lượng chứng khoán niêm yết (3)"]))
        ws.cell(row=excel_row, column=6, value=num_or_dash(row["Vốn chủ sở hữu của CTCK (4)"]))

        # 2 cột TỶ LỆ: GIỮ LÀ SỐ, để Excel format dạng %
        v1 = row["Tỷ lệ dư nợ/VCSH (1)/(4)"]
        v2 = row["Tỷ lệ CK cho vay/CKNY (2)/(3)"]
        ws.cell(
            row=excel_row,
            column=7,
            value=float(v1) if pd.notna(v1) else None,
        )
        ws.cell(
            row=excel_row,
            column=8,
            value=float(v2) if pd.notna(v2) else None,
        )

    # 8. Hàng "Tổng cộng" – ngay dưới các mã
    total_row     = first_data_row + len(df)
    last_data_row = first_data_row + len(df) - 1   # dòng dữ liệu cuối cùng

    total_debt = df["Dư nợ cho vay GDKQ (1)"].sum()
    total_qty  = df["Số lượng chứng khoán cho vay (2)"].sum()

    ws.cell(row=total_row, column=2, value="Tổng cộng")
    ws.cell(row=total_row, column=3, value=num_or_dash(total_debt))
    ws.cell(row=total_row, column=4, value=num_or_dash(total_qty))

    # =========================
    # 9. Format giống báo cáo ngày
    # =========================

    # Style cơ bản
    title_font   = Font(bold=True, size=16)
    unit_font    = Font(italic=True, size=13)
    header_font  = Font(bold=True, color="FFFFFF")  # chữ trắng
    header_fill  = PatternFill(start_color="008000", end_color="008000", fill_type="solid")  # xanh lá đậm
    thin         = Side(border_style="thin", color="000000")
    border       = Border(left=thin, right=thin, top=thin, bottom=thin)
    center       = Alignment(horizontal="center", vertical="center", wrap_text=True)
    right_align  = Alignment(horizontal="right", vertical="center")

    # Tiêu đề
    ws["A1"].font      = title_font
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws["A2"].font      = unit_font
    ws["A2"].alignment = Alignment(horizontal="center", vertical="center")

    # Header row style (hàng 3)
    for col_idx in range(1, n_cols + 1):
        cell = ws.cell(row=3, column=col_idx)
        cell.font      = header_font
        cell.fill      = header_fill
        cell.alignment = center
        cell.border    = border

    # Kẻ bảng (border) cho header + các dòng dữ liệu (KHÔNG kẻ dòng Tổng cộng)
    for row in ws.iter_rows(
        min_row=3,
        max_row=last_data_row,
        min_col=1,
        max_col=n_cols,
    ):
        for cell in row:
            cell.border = border

    # Border riêng cho dòng 'Tổng cộng' – chỉ kẻ ở cột 2 → 4
    for col_idx in range(2, 5):  # B, C, D
        ws.cell(row=total_row, column=col_idx).border = border

    # In đậm CHỈ các số tổng ở cột 3 và 4
    bold_font = Font(bold=True)
    for col_idx in (3, 4):
        ws.cell(row=total_row, column=col_idx).font = bold_font

    # Number format cho các cột số (bao gồm hàng tổng)
    for row in ws.iter_rows(min_row=4, max_row=total_row, min_col=1, max_col=n_cols):
        # Dư nợ (1) – cột 3
        row[2].number_format = "#,##0"
        # Số lượng CK cho vay (2), niêm yết (3) – cột 4,5
        row[3].number_format = "#,##0"
        row[4].number_format = "#,##0"
        # VCSH – cột 6
        row[5].number_format = "#,##0"
        # Tỷ lệ – cột 7,8 (%)
        row[6].number_format = "0.00%"
        row[7].number_format = "0.00%"

    # Canh phải các cột số (3–8), gồm cả dấu "-"
    for row in ws.iter_rows(
        min_row=4,
        max_row=total_row,
        min_col=3,
        max_col=8,
    ):
        for cell in row:
            cell.alignment = right_align

    # Set width cột
    width_map = {
        1: 5,   # STT
        2: 25,  # Mã CK
        3: 25,  # Dư nợ
        4: 25,  # SL cho vay
        5: 25,  # SL niêm yết
        6: 25,  # VCSH
        7: 25,  # Tỷ lệ dư nợ/VCSH
        8: 25,  # Tỷ lệ CK cho vay/CKNY
    }
    for col_idx, width in width_map.items():
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = width

    # Freeze panes: cố định tiêu đề + header
    ws.freeze_panes = "A4"

    # 10. Lưu file
    wb.save(output_report_path)
    print(f"Đã tạo file báo cáo ngày: {output_report_path}")


# =========================
# 3. Chạy thử
# =========================

if __name__ == "__main__":
    RAW_FILE          = r"180426__RMD_SCMS_Bao cao ngay truoc 8AM 01122025 New check- Dữ liệu raw.xlsx"
    SPECIAL_ROOM_FILE = r"1 số room đặc biệt.xlsx"

    # File 1: copy raw + thêm sheet kỹ thuật
    RAW_OUTPUT = r"180426__RMD_SCMS_Bao cao ngay truoc 8AM 01122025_New_PY.xlsx"
    write_margin_report_sheet(
        excel_input_path=RAW_FILE,
        excel_output_path=RAW_OUTPUT,
        sheet_name="Dư nợ theo used room_PY",
        equity_vnd_million=2_077_000.0,
        special_room_path=SPECIAL_ROOM_FILE,
    )

    # File 2: file Báo cáo ngày để dùng thực tế
    REPORT_FILE = r"Báo cáo ngày 2025.12.01_PY.xlsx"
    write_margin_report_file(
        raw_excel_path=RAW_FILE,
        output_report_path=REPORT_FILE,
        sheet_name="Report",
        equity_vnd_million=2_077_000.0,
        special_room_path=SPECIAL_ROOM_FILE,
    )
