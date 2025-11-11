# -*- coding: utf-8 -*-
"""
02_od_assign_cn_v3.py

阶段一：跨公司调拨优化（增强版，线性近似整车成本 + 小批量惩罚）
 - Step-1: 最小化欠配
 - Step-2: 固定最小欠配下，最小化带小批量惩罚的单位代理成本
 - 允许运输日期弹性（基于路网时长离散）
 - 产出：流向方案_LP.csv、未满足统计.csv
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import pulp as pl

# ---------------- 全局默认参数 ----------------
VEHICLE_CAPACITY = 16000           # 车辆容量 C（羽）
RHO_BASE = 0.000176                # 基础 单羽·公里 代理价
ALPHA_PENALTY = 1.0                # 小批量惩罚系数 α，越大越惩罚小批
Q_MIN_HINT = 8000                  # 参考批量下限 Q_min（避免目的地极小日被过度惩罚）


# ---------------- 工具函数 ----------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.map(lambda x: str(x).strip().replace(" ", "").replace("　", ""))
    return df


def _map_network_columns(net: pd.DataFrame) -> pd.DataFrame:
    net = _clean_columns(net)
    mapping = {
        "src_company": "出发地", "source": "出发地", "src": "出发地", "from": "出发地",
        "dst_company": "到达地", "destination": "到达地", "dst": "到达地", "to": "到达地",
        "dist_km": "运输距离", "distance_km": "运输距离", "distance": "运输距离", "里程": "运输距离",
        "time_h": "运输时长", "time_hr": "运输时长", "duration_h": "运输时长"
    }
    net.rename(columns=mapping, inplace=True)
    return net


def _transport_day_options(hours: float) -> Tuple[int, int]:
    """
    将运输时长(小时)离散为两个运输天数候选：
      <=24h → {0,1}; 24<h<=48 → {1,2}; 以此类推
    """
    if pd.isna(hours):
        return (0, 1)
    h = float(hours)
    if h <= 24:
        return (0, 1)
    floor_day = int(h // 24)
    return (floor_day, floor_day + 1)


def _age_penalty(age_bin: str, age_penalty_le25: float) -> float:
    return 0.0 if str(age_bin).upper() == "GT25" else float(age_penalty_le25)


# ---------------- 读取与清洗 ----------------
def read_inputs(data_dir: str, out_dir: str, start_date: str, end_date: str):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    sup = _clean_columns(pd.read_csv(out_dir / "剩余供给.csv"))
    dem = _clean_columns(pd.read_csv(out_dir / "剩余需求.csv"))

    # 路网：支持 network.csv 或 路网.csv
    net_candidates = [data_dir / "路网.csv", data_dir / "network.csv"]
    net_path = next((p for p in net_candidates if p.exists()), None)
    if net_path is None:
        raise FileNotFoundError(f"缺少路网文件：{net_candidates[0]} 或 {net_candidates[1]}")
    net = _map_network_columns(pd.read_csv(net_path))

    # 必要列检查
    need_sup = ["日期", "组织单元", "品种", "性别", "周龄档", "数量"]
    miss = [c for c in need_sup if c not in sup.columns]
    if miss:
        raise ValueError(f"剩余供给.csv 缺列：{miss}，现有：{list(sup.columns)}")

    need_dem = ["日期", "组织单元", "品种", "性别", "数量"]
    miss = [c for c in need_dem if c not in dem.columns]
    if miss:
        raise ValueError(f"剩余需求.csv 缺列：{miss}，现有：{list(dem.columns)}")

    need_net = ["出发地", "到达地", "运输距离"]
    miss = [c for c in need_net if c not in net.columns]
    if miss:
        raise ValueError(f"{net_path.name} 缺列：{miss}，现有：{list(net.columns)}")

    # 类型/清洗
    for df in (sup, dem):
        for c in ["组织单元", "品种", "性别", "周龄档"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        df["数量"] = pd.to_numeric(df["数量"], errors="coerce").fillna(0).astype(int)

    # 日期
    for df, col in [(sup, "日期"), (dem, "日期")]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    s, e = pd.to_datetime(start_date), pd.to_datetime(end_date)
    sup = sup[(sup["日期"] >= s) & (sup["日期"] <= e)].copy()
    dem = dem[(dem["日期"] >= s) & (dem["日期"] <= e)].copy()

    # 路网：双向补全 + 天数候选
    net["出发地"] = net["出发地"].astype(str).str.strip()
    net["到达地"] = net["到达地"].astype(str).str.strip()
    net["运输距离"] = pd.to_numeric(net["运输距离"], errors="coerce").astype(float)
    if "运输时长" in net.columns:
        net["运输时长"] = pd.to_numeric(net["运输时长"], errors="coerce")

    dist_map: Dict[Tuple[str, str], float] = {}
    day_options: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for _, r in net.iterrows():
        i, j = r["出发地"], r["到达地"]
        if not i or not j:
            continue
        dist_map[(i, j)] = float(r["运输距离"])
        dist_map[(j, i)] = float(r["运输距离"])
        dopt = _transport_day_options(r["运输时长"] if "运输时长" in r else None)
        day_options[(i, j)] = dopt
        day_options[(j, i)] = dopt

    return sup, dem, dist_map, day_options, s, e


# ---------------- LP 构建与求解 ----------------
def build_and_solve_lp(
    sup: pd.DataFrame,
    dem: pd.DataFrame,
    dist_map: dict,
    day_options: dict,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rho: float,
    alpha_penalty: float,
    age_penalty_le25: float,
    small_threshold: int,
    vehicle_capacity: int,
    q_min_hint: int,
):
    """
    两阶段：Step-1 最小欠配；Step-2 在固定欠配下最小成本（带小批量惩罚的线性近似）
    """

    # —— 供需映射 —— #
    S = {}  # (ship_d, i, p, s, a) -> qty
    for _, r in sup.iterrows():
        key = (r["日期"], r["组织单元"], r["品种"], r["性别"], r["周龄档"])
        S[key] = S.get(key, 0) + int(r["数量"])

    R = {}  # (arr_d, j, p, s) -> qty
    skipped_small = []  # 被小单阈值剔除的整条需求
    for _, r in dem.iterrows():
        key = (r["日期"], r["组织单元"], r["品种"], r["性别"])
        qty = int(r["数量"])
        if qty < small_threshold:
            skipped_small.append((key, qty))
        else:
            R[key] = R.get(key, 0) + qty

    # —— 目的地-到达日 参考批量 Q_{t,j} —— #
    # 用原始 dem（未过阈值）聚合更贴近真实：sum_{p,s} R_{t,j,p,s}，并夹在 [q_min_hint, C]
    dem_day_dst = (dem.groupby(["日期", "组织单元"], as_index=False)["数量"].sum())
    ref_Q = {}  # (arr_d, j) -> Q_tj
    for _, r in dem_day_dst.iterrows():
        arr_d = pd.to_datetime(r["日期"])
        j = r["组织单元"]
        q = int(r["数量"])
        q = max(q_min_hint, min(q, vehicle_capacity))
        ref_Q[(arr_d, j)] = q

    # —— 变量索引（只为可行索引建变量） —— #
    x_index = []  # (ship_d, arr_d, i, j, p, s, a)
    for (ship_d, i, p, s, a), _sqty in S.items():
        for (arr_d, j, pp, ss), _rqty in R.items():
            if pp != p or ss != s:
                continue
            if (i, j) not in dist_map:
                continue
            d1, d2 = day_options.get((i, j), (0, 1))
            max_days = int(math.ceil(d2))
            # 只要在 0..max_days 的窗口里都允许（更宽松，避免被定死）
            if 0 <= (arr_d - ship_d).days <= max_days:
                x_index.append((ship_d, arr_d, i, j, p, s, a))

    u_index = list(R.keys())  # 欠配变量键

    # ========= Step-1：最小欠配 ========= #
    m1 = pl.LpProblem("Step1_MinUnmet", pl.LpMinimize)
    x = pl.LpVariable.dicts("x", x_index, lowBound=0)
    u = pl.LpVariable.dicts("u", u_index, lowBound=0)

    m1 += pl.lpSum(u[k] for k in u_index)

    # 供给约束
    for key, sval in S.items():
        ship_d, i, p, s, a = key
        terms = [
            x[idx] for idx in x_index
            if idx[0] == ship_d and idx[2] == i and idx[4] == p and idx[5] == s and idx[6] == a
        ]
        if terms:
            m1 += pl.lpSum(terms) <= sval

    # 需求约束
    for key, rqty in R.items():
        arr_d, j, p, s = key
        terms = [
            x[idx] for idx in x_index
            if idx[1] == arr_d and idx[3] == j and idx[4] == p and idx[5] == s
        ]
        if terms:
            m1 += pl.lpSum(terms) + u[key] == rqty
        else:
            m1 += u[key] == rqty

    m1.solve(pl.PULP_CBC_CMD(msg=False))
    min_unmet = pl.value(pl.lpSum(u[k] for k in u_index)) or 0.0

    # ========= Step-2：固定最小欠配下的最小成本 ========= #
    m2 = pl.LpProblem("Step2_MinCost", pl.LpMinimize)
    x2 = pl.LpVariable.dicts("x", x_index, lowBound=0)
    u2 = pl.LpVariable.dicts("u", u_index, lowBound=0)

    # 单位代理成本（线性近似整车 + 小批量惩罚）
    # cost = dist * rho * (1 + alpha * (C - Q_tj)/C) + age_penalty
    def unit_proxy(idx) -> float:
        ship_d, arr_d, i, j, p, s, a = idx
        dist = dist_map[(i, j)]
        Q_tj = ref_Q.get((arr_d, j), vehicle_capacity)  # 没有就取 C（即不惩罚）
        penalty = 1.0 + alpha_penalty * (vehicle_capacity - Q_tj) / float(vehicle_capacity)
        base = dist * rho * penalty
        return base + _age_penalty(a, age_penalty_le25)

    m2 += pl.lpSum(x2[idx] * unit_proxy(idx) for idx in x_index)

    # 供给约束
    for key, sval in S.items():
        ship_d, i, p, s, a = key
        terms = [
            x2[idx] for idx in x_index
            if idx[0] == ship_d and idx[2] == i and idx[4] == p and idx[5] == s and idx[6] == a
        ]
        if terms:
            m2 += pl.lpSum(terms) <= sval

    # 需求约束
    for key, rqty in R.items():
        arr_d, j, p, s = key
        terms = [
            x2[idx] for idx in x_index
            if idx[1] == arr_d and idx[3] == j and idx[4] == p and idx[5] == s
        ]
        if terms:
            m2 += pl.lpSum(terms) + u2[key] == rqty
        else:
            m2 += u2[key] == rqty

    # 固定最小欠配
    m2 += pl.lpSum(u2[k] for k in u_index) == min_unmet

    m2.solve(pl.PULP_CBC_CMD(msg=False))

    # ========= 结果收集 ========= #
    raw_rows = []
    for idx in x_index:
        val = x2[idx].value()
        if val and val > 1e-6:
            ship_d, arr_d, i, j, p, s, a = idx
            dist = dist_map[(i, j)]
            # 这里的“单位代理成本”仅展示基础 rho*dist，便于直观对比；
            # 真正用于优化的单位成本包含了小批量惩罚和年龄惩罚（已体现在解里）。
            raw_rows.append({
                "日期": pd.to_datetime(ship_d).strftime("%Y-%m-%d"),   # 发车日
                "到达日期": pd.to_datetime(arr_d).strftime("%Y-%m-%d"),
                "出发地": i, "到达地": j,
                "品种": p, "性别": s, "周龄档": a,
                "数量": float(val),
                "运输距离": float(dist),
                "单位代理成本": float(dist * rho),
                "代理总成本": float(dist * rho * val),
            })

    flow_df = pd.DataFrame(raw_rows, columns=[
        "日期", "到达日期", "出发地", "到达地", "品种", "性别", "周龄档",
        "数量", "运输距离", "单位代理成本", "代理总成本"
    ])

    # ========= 未满足统计 ========= #
    unmet_rows = []

    # 1) 小单阈值剔除的整条需求
    for (arr_d, j, p, s), qty in skipped_small:
        unmet_rows.append({
            "日期": pd.to_datetime(arr_d).strftime("%Y-%m-%d"),
            "到达地": j, "品种": p, "性别": s,
            "未满足数量": int(qty),
            "未满足原因": "小于阈值的需求量"
        })

    # 2) LP 未满足（按旬初/旬中）——若需要更细分（无该品类/无法按时到达），可加可行性检测
    for key in u_index:
        val = u2[key].value()
        if val and val > 1e-6:
            arr_d, j, p, s = key
            reason = "旬初供给不足" if pd.to_datetime(arr_d) == start_date else "旬中供给不足"
            unmet_rows.append({
                "日期": pd.to_datetime(arr_d).strftime("%Y-%m-%d"),
                "到达地": j, "品种": p, "性别": s,
                "未满足数量": int(round(val)),
                "未满足原因": reason
            })

    # ========= 发运阈值聚合过滤（抑制碎片流） ========= #
    # 同一(到达日期, 到达地, 品种, 性别, 周龄档, 出发地, 发车日) 聚合后，小于 small_threshold 的批次不发，计入未满足
    if not flow_df.empty:
        group_cols = ["到达日期", "到达地", "品种", "性别", "周龄档", "出发地", "日期"]
        agg = (flow_df.groupby(group_cols, as_index=False)["数量"].sum())

        small_mask = agg["数量"] < float(small_threshold)
        small_batches = agg[small_mask].copy()
        kept_batches = agg[~small_mask].copy()

        # 小批次 → 不发，计入未满足
        if not small_batches.empty:
            for _, r in small_batches.iterrows():
                unmet_rows.append({
                    "日期": r["到达日期"],
                    "到达地": r["到达地"],
                    "品种": r["品种"],
                    "性别": r["性别"],
                    "未满足数量": int(round(r["数量"])),
                    "未满足原因": "没有满足最小运输阈值"
                })

        # 保留的大批次：补齐成本列并输出
        if kept_batches.empty:
            flow_df = pd.DataFrame(columns=[
                "日期", "到达日期", "出发地", "到达地", "品种", "性别", "周龄档",
                "数量", "运输距离", "单位代理成本", "代理总成本"
            ])
        else:
            def _unit_cost(row):
                i, j, age = row["出发地"], row["到达地"], row["周龄档"]
                dist_km = dist_map[(i, j)]
                unit_cost = dist_km * rho + _age_penalty(age, age_penalty_le25)
                return dist_km, unit_cost

            costs = kept_batches.apply(_unit_cost, axis=1, result_type="expand")
            kept_batches["运输距离"] = costs[0].astype(float)
            kept_batches["单位代理成本"] = costs[1].astype(float)
            kept_batches["数量"] = kept_batches["数量"].round().astype(int)
            kept_batches["代理总成本"] = kept_batches["单位代理成本"] * kept_batches["数量"]

            flow_df = kept_batches[[
                "日期", "到达日期", "出发地", "到达地", "品种", "性别", "周龄档",
                "数量", "运输距离", "单位代理成本", "代理总成本"
            ]].copy()

    unmet_df = pd.DataFrame(unmet_rows, columns=["日期", "到达地", "品种", "性别", "未满足数量", "未满足原因"])
    if not unmet_df.empty:
        unmet_df["未满足数量"] = pd.to_numeric(unmet_df["未满足数量"], errors="coerce").fillna(0).astype(int)
        unmet_df = (unmet_df
                    .groupby(["日期", "到达地", "品种", "性别", "未满足原因"], as_index=False)["未满足数量"]
                    .sum())

    return flow_df, unmet_df


# ---------------- 主程序 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./data_out")
    ap.add_argument("--start_date", type=str, default="2025-07-01")
    ap.add_argument("--end_date", type=str, default="2025-07-31")
    ap.add_argument("--rho", type=float, default=RHO_BASE, help="元/(羽·km)")
    ap.add_argument("--alpha", type=float, default=ALPHA_PENALTY, help="小批量惩罚系数 α（0.6~1.2 建议）")
    ap.add_argument("--age_penalty_le25", type=float, default=0.05, help="LE25 年龄惩罚（元/羽）")
    ap.add_argument("--small_threshold", type=int, default=200, help="最小运输阈值/小单阈值（羽）")
    ap.add_argument("--vehicle_capacity", type=int, default=VEHICLE_CAPACITY, help="车辆容量 C（羽）")
    ap.add_argument("--q_min_hint", type=int, default=Q_MIN_HINT, help="目的地-到达日参考批量下限（羽）")
    args = ap.parse_args()

    sup, dem, dist_map, day_options, s, e = read_inputs(
        args.data_dir, args.out_dir, args.start_date, args.end_date
    )

    flow_df, unmet_df = build_and_solve_lp(
        sup, dem, dist_map, day_options, s, e,
        rho=args.rho,
        alpha_penalty=args.alpha,
        age_penalty_le25=args.age_penalty_le25,
        small_threshold=args.small_threshold,
        vehicle_capacity=args.vehicle_capacity,
        q_min_hint=args.q_min_hint,
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    flow_df = flow_df.sort_values(["到达日期", "出发地", "到达地", "品种", "性别", "周龄档", "日期"])
    flow_path = out_dir / "流向方案_LP.csv"
    flow_df.to_csv(flow_path, index=False, encoding="utf-8-sig")
    print(f"✅ 流向方案_LP.csv: {len(flow_df)} 行 → {flow_path}")

    if not unmet_df.empty:
        unmet_path = out_dir / "未满足统计.csv"
        unmet_df = unmet_df.sort_values(["日期", "到达地", "品种", "性别", "未满足原因"])
        unmet_df.to_csv(unmet_path, index=False, encoding="utf-8-sig")
        print(f"⚠️ 未满足统计.csv: {len(unmet_df)} 行 → {unmet_path}")
    else:
        print("✅ 无欠配（包含小单过滤之后）")


if __name__ == "__main__":
    main()
