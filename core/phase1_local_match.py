# -*- coding: utf-8 -*-
"""
01_local_match_cn.py
功能：
  阶段一：公司内部本地自配（组织单元相同）
  - 输入：supply.csv（出孵/出雏数据），demand.csv（投苗数据）
  - 逻辑：对每个【日期×组织单元×品种×性别】的需求：
      1) 先用 GT25（>25周龄），不足再用 LE25（≤25周龄）；
      2) 若本地未满足量在「需求量的 R%（默认5%）」之内，视为本地自动补齐，不进入跨公司调拨，
         并在《当地流动.csv》用“阈值内缺口”标记该值，同时“数量”显示为原始需求量（=已配 + 缺口）。
      3) 超过比例阈值部分，进入《剩余需求.csv》。
  - 输出：
      - 当地流动.csv  —— 键级（日期、出发地=到达地、品种、性别）聚合后的本地满足结果，含“阈值内缺口”
      - 剩余供给.csv  —— 未被本地需求吸收的供给（=原始供给 − 实际本地发出）
      - 剩余需求.csv  —— 本地未满足且超过比例阈值的需求部分
"""

import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd


# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------
def normalize_str(x):
    """字符串去空白；缺失则保持"""
    if pd.isna(x):
        return x
    return str(x).strip()


def ensure_age_bin(df, age_threshold=25):
    """
    若供给表未提供“周龄档”，则依据“周龄”字段自动生成：
      周龄 <= 25 → LE25；周龄 > 25 → GT25
    """
    df = df.copy()
    if "周龄档" not in df.columns:
        if "周龄" not in df.columns:
            raise ValueError("供给表缺少【周龄档】或【周龄】，无法生成周龄分档。")
        df["周龄档"] = df["周龄"].astype(float).apply(
            lambda a: "LE25" if a <= age_threshold else "GT25"
        )
    return df


# ---------------------------------------------------------------------
# 读取输入
# ---------------------------------------------------------------------
def read_inputs(data_dir, start_date, end_date):
    """
    读取 supply.csv / demand.csv 并清洗；支持 出孵/出雏 两种写法。
    """
    data_dir = Path(data_dir)
    supply = pd.read_csv(data_dir / "supply.csv")
    demand = pd.read_csv(data_dir / "demand.csv")

    # 列名去空白
    supply.columns = supply.columns.map(lambda x: str(x).strip().replace(" ", "").replace("　", ""))
    demand.columns = demand.columns.map(lambda x: str(x).strip().replace(" ", "").replace("　", ""))

    # 中文列名映射（兼容“出孵/出雏”）
    supply.rename(columns={
        "出孵日期": "日期", "出孵数量": "数量",
        "出雏日期": "日期", "出雏数量": "数量",
        "组织单元": "组织单元", "品种": "品种", "饲养组合": "饲养组合"
    }, inplace=True)
    demand.rename(columns={
        "投苗日期": "日期", "投苗数量": "数量",
        "组织单元": "组织单元", "品种": "品种", "饲养组合": "饲养组合"
    }, inplace=True)

    # 性别映射：母/公/混 → F/M/MIX
    def normalize_sex(s):
        mapping = {"母": "F", "雌": "F", "公": "M", "雄": "M", "混": "MIX"}
        return mapping.get(str(s).strip(), str(s).strip().upper())

    supply["性别"] = supply["饲养组合"].map(normalize_sex)
    demand["性别"] = demand["饲养组合"].map(normalize_sex)

    # 日期过滤
    supply["日期"] = pd.to_datetime(supply["日期"])
    demand["日期"] = pd.to_datetime(demand["日期"])
    s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
    supply = supply[(supply["日期"] >= s) & (supply["日期"] <= e)].copy()
    demand = demand[(demand["日期"] >= s) & (demand["日期"] <= e)].copy()

    # 数量清洗
    supply["数量"] = pd.to_numeric(supply["数量"], errors="coerce").fillna(0).astype(int)
    demand["数量"] = pd.to_numeric(demand["数量"], errors="coerce").fillna(0).astype(int)
    supply = supply[supply["数量"] > 0].copy()
    demand = demand[demand["数量"] > 0].copy()

    # 周龄档
    supply = ensure_age_bin(supply)

    return supply, demand


# ---------------------------------------------------------------------
# 本地自配 + 比例阈值内缺口补齐
# ---------------------------------------------------------------------
def local_match_with_ratio_threshold(supply, demand, gap_ratio=0.05):
    """
    对每个键 (日期, 组织单元, 品种, 性别)：
      1) 先用 GT25，再用 LE25；
      2) 若本地剩余需求 gap ∈ (0, demand*gap_ratio]，记为“阈值内缺口”（自动补齐，不消耗供给），
         并在《当地流动》按键级显示“数量=原始需求量、阈值内缺口=gap”；
      3) 若 gap > demand*gap_ratio，则剩余 gap 进入《剩余需求》。
    备注：剩余供给 = 原始供给 − 实际本地发出（阈值内缺口不消耗供给）。
    """

    # —— 1) 汇总供给与需求（到键）
    sup_gt = (supply[supply["周龄档"] == "GT25"]
              .groupby(["日期", "组织单元", "品种", "性别"], as_index=False)["数量"].sum())
    sup_le = (supply[supply["周龄档"] == "LE25"]
              .groupby(["日期", "组织单元", "品种", "性别"], as_index=False)["数量"].sum())
    dem_sum = (demand.groupby(["日期", "组织单元", "品种", "性别"], as_index=False)["数量"].sum())

    # —— 2) 字典化（便于扣减）
    def to_dict(df):
        return {(r["日期"], r["组织单元"], r["品种"], r["性别"]): int(r["数量"])
                for _, r in df.iterrows()}

    rem_gt = defaultdict(int, to_dict(sup_gt))   # 剩余 GT25 供给
    rem_le = defaultdict(int, to_dict(sup_le))   # 剩余 LE25 供给
    rem_dem = defaultdict(int, to_dict(dem_sum)) # 剩余需求（动态扣减）
    demand_ref = to_dict(dem_sum)                # 原始需求快照（计算比例阈值用）

    local_rows = []  # 本地配明细（GT25/LE25 + 阈值补齐行）

    # —— 3) 匹配（先GT25后LE25）
    for key in list(rem_dem.keys()):
        need = rem_dem.get(key, 0)
        if need <= 0:
            continue

        # 先用 GT25
        take_gt = min(rem_gt.get(key, 0), need)
        if take_gt > 0:
            rem_gt[key] -= take_gt
            rem_dem[key] -= take_gt
            d, org, prod, sex = key
            local_rows.append({
                "日期": d.date() if isinstance(d, pd.Timestamp) else d,
                "出发地": org, "到达地": org, "品种": prod, "性别": sex,
                "周龄档": "GT25", "数量": int(take_gt),
                "运输距离": 0, "运输成本": 0.0, "调度标识": 0,
                "阈值内缺口": 0
            })

        # 再用 LE25
        remain = rem_dem.get(key, 0)
        take_le = min(rem_le.get(key, 0), remain)
        if take_le > 0:
            rem_le[key] -= take_le
            rem_dem[key] -= take_le
            d, org, prod, sex = key
            local_rows.append({
                "日期": d.date() if isinstance(d, pd.Timestamp) else d,
                "出发地": org, "到达地": org, "品种": prod, "性别": sex,
                "周龄档": "LE25", "数量": int(take_le),
                "运输距离": 0, "运输成本": 0.0, "调度标识": 0,
                "阈值内缺口": 0
            })

        # —— 4) 比例阈值判定：gap <= demand * gap_ratio → 记为阈值内缺口（不进剩余需求）
        gap = rem_dem.get(key, 0)
        if gap > 0:
            orig_dem = demand_ref.get(key, 0)
            # 若原始需求为0（理论不该出现），则不触发比例补齐
            if orig_dem > 0 and gap <= orig_dem * gap_ratio:
                d, org, prod, sex = key
                local_rows.append({
                    "日期": d.date() if isinstance(d, pd.Timestamp) else d,
                    "出发地": org, "到达地": org, "品种": prod, "性别": sex,
                    "周龄档": "阈值补齐", "数量": int(gap),
                    "运输距离": 0, "运输成本": 0.0, "调度标识": 0,
                    "阈值内缺口": int(gap)
                })
                rem_dem[key] = 0  # 阈值内缺口不再进入剩余需求

    # —— 5) 生成《当地流动.csv》（键级聚合；数量显示为原始需求量）
    local_df = pd.DataFrame(local_rows, columns=[
        "日期", "出发地", "到达地", "品种", "性别", "周龄档",
        "数量", "运输距离", "运输成本", "调度标识", "阈值内缺口"
    ])

    if not local_df.empty:
        key_cols = ["日期", "出发地", "到达地", "品种", "性别"]
        agg_local = (local_df.groupby(key_cols, as_index=False)[["数量", "阈值内缺口"]]
                     .sum())

        # 将数量修正为“原始需求量”
        rows_fixed = []
        for _, r in agg_local.iterrows():
            key = (pd.to_datetime(r["日期"]), r["出发地"], r["品种"], r["性别"])
            # 容错：键中“出发地=到达地”，两者等价
            if key not in demand_ref:
                key = (pd.to_datetime(r["日期"]), r["到达地"], r["品种"], r["性别"])
            dem_val = demand_ref.get(key, int(r["数量"]))
            rf = r.copy()
            rf["数量"] = int(dem_val)
            rows_fixed.append(rf)

        local_flow = pd.DataFrame(rows_fixed)[key_cols + ["数量", "阈值内缺口"]]
        local_flow["运输距离"] = 0
        local_flow["运输成本"] = 0.0
        local_flow["调度标识"] = 0
        local_flow = local_flow[[
            "日期", "出发地", "到达地", "品种", "性别",
            "数量", "阈值内缺口", "运输距离", "运输成本", "调度标识"
        ]].sort_values(["日期", "出发地", "品种", "性别"])
    else:
        local_flow = pd.DataFrame(columns=[
            "日期", "出发地", "到达地", "品种", "性别",
            "数量", "阈值内缺口", "运输距离", "运输成本", "调度标识"
        ])

    # —— 6) 生成《剩余供给.csv》（仍按周龄档还原；= 原始供给 − 实际本地发出）
    res_sup = []
    for key, q in rem_gt.items():
        if q > 0:
            res_sup.append([key[0], key[1], key[2], key[3], "GT25", int(q)])
    for key, q in rem_le.items():
        if q > 0:
            res_sup.append([key[0], key[1], key[2], key[3], "LE25", int(q)])
    residual_supply = pd.DataFrame(
        res_sup, columns=["日期", "组织单元", "品种", "性别", "周龄档", "数量"]
    ).sort_values(["日期", "组织单元", "品种", "性别", "周龄档"])

    # —— 7) 生成《剩余需求.csv》（仅保留 > 比例阈值 的缺口）
    res_dem = []
    for key, q in rem_dem.items():
        if q > 0:
            res_dem.append([key[0], key[1], key[2], key[3], int(q)])
    residual_demand = pd.DataFrame(
        res_dem, columns=["日期", "组织单元", "品种", "性别", "数量"]
    ).sort_values(["日期", "组织单元", "品种", "性别"])

    return local_flow, residual_supply, residual_demand


# ---------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="输入数据目录")
    parser.add_argument("--out_dir", type=str, default="./data_out", help="输出目录")
    parser.add_argument("--start_date", type=str, default="2025-07-01", help="开始日期")
    parser.add_argument("--end_date", type=str, default="2025-07-31", help="结束日期")
    parser.add_argument("--age_threshold", type=int, default=25, help="周龄分档阈值")
    parser.add_argument("--local_gap_ratio", type=float, default=0.05,
                        help="本地缺口比例阈值（例如 0.05 表示 5%）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    supply, demand = read_inputs(args.data_dir, args.start_date, args.end_date)
    local_flow, residual_supply, residual_demand = local_match_with_ratio_threshold(
        supply, demand, gap_ratio=args.local_gap_ratio
    )

    local_flow.to_csv(out_dir / "当地流动.csv", index=False, encoding="utf-8-sig")
    residual_supply.to_csv(out_dir / "剩余供给.csv", index=False, encoding="utf-8-sig")
    residual_demand.to_csv(out_dir / "剩余需求.csv", index=False, encoding="utf-8-sig")

    print(f"✅ 当地流动.csv: {len(local_flow)} 条记录")
    print(f"✅ 剩余供给.csv: {len(residual_supply)} 条记录")
    print(f"✅ 剩余需求.csv: {len(residual_demand)} 条记录")


if __name__ == "__main__":
    main()
