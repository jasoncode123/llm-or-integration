# -*- coding: utf-8 -*-
"""
02_od_assign_cn_v3.py  (定死发/到日 + 拼车; 成本按车汇总)

派车（拼车+整车，整车计费，区域车池+冷却，当天可复用）：
- 输入：
  data_out/流向方案_LP.csv : 日期(发车日),到达日期(到达日),出发地,到达地,品种,性别,周龄档,数量
  data/network.csv 或 data/路网.csv : src_company,dst_company,dist_km,time_h
  data/fleet.csv : 车牌号,是否可用,核载数量,长度规格,承包区域,初始位置(可空)
- 输出：
  data_out/派车计划.csv                (发车日,到达日,出发地,到达地,车牌号,区域ID,品种,性别,数量)
  data_out/派车总成本汇总.csv          (按“车次”计费后，再按发车日汇总 + TOTAL)
  data_out/未满足统计_派车阶段.csv      (包含品种、性别)
  data_out/cbc_log.txt                 (CBC 求解进度)
"""
import argparse, math, datetime
from pathlib import Path
from collections import defaultdict
import pandas as pd
import pulp as pl

# ---------- 可调参数 ----------
EPS = 1e-6
TIME_LIMIT_SEC = 600
PENALTY_UNMET = 1e6      # 欠配惩罚（远大于任何一趟车成本）
MIN_BATCH = 1            # 激活车次后最小装载，避免 L=0

# 仅按装载量分段计价（lower, upper, unit_price_per_km）
BRACKETS = [
    (0,     16000, 6.1),
    (16001, 20000, 7.1),
    (20001, 30000, 8.4),
    (30001, 50000, 8.8),
]

# ---------- 工具函数 ----------
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "").replace("　", "") for c in df.columns]
    return df

def _read_flow(out_dir: Path) -> pd.DataFrame:
    fp = out_dir / "流向方案_LP.csv"
    if not fp.exists(): raise FileNotFoundError(f"缺少 {fp}")
    df = _clean_cols(pd.read_csv(fp))
    need = ["日期","到达日期","出发地","到达地","品种","性别","周龄档","数量"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"流向方案_LP.csv 缺列：{miss}")
    df["数量"] = pd.to_numeric(df["数量"], errors="coerce").fillna(0.0)
    df = df[df["数量"] > 0].copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")         # 固定为发车日
    df["到达日期"] = pd.to_datetime(df["到达日期"], errors="coerce")   # 固定为到达日
    return df

def _read_network(data_dir: Path) -> pd.DataFrame:
    for name in ["路网.csv","network.csv"]:
        p = data_dir / name
        if p.exists():
            net = _clean_cols(pd.read_csv(p))
            mapping = {
                "src_company":"出发地","source":"出发地","src":"出发地","from":"出发地",
                "dst_company":"到达地","destination":"到达地","dst":"到达地","to":"到达地",
                "dist_km":"运输距离","distance_km":"运输距离","distance":"运输距离","里程":"运输距离",
                "time_h":"运输时长","time_hr":"运输时长","duration_h":"运输时长"
            }
            net.rename(columns={k:v for k,v in mapping.items() if k in net.columns}, inplace=True)
            need = ["出发地","到达地","运输距离"]
            miss = [c for c in need if c not in net.columns]
            if miss: raise ValueError(f"{name} 缺列：{miss}")
            net["出发地"] = net["出发地"].astype(str).str.strip()
            net["到达地"] = net["到达地"].astype(str).str.strip()
            net["运输距离"] = pd.to_numeric(net["运输距离"], errors="coerce").astype(float)
            if "运输时长" in net.columns:
                net["运输时长"] = pd.to_numeric(net["运输时长"], errors="coerce")
            else:
                net["运输时长"] = float("nan")
            return net
    raise FileNotFoundError("data/路网.csv 或 data/network.csv 未找到")

def _read_fleet(data_dir: Path) -> pd.DataFrame:
    fp = data_dir / "fleet.csv"
    if not fp.exists(): raise FileNotFoundError(f"缺少 {fp}")
    fl = _clean_cols(pd.read_csv(fp))
    mapping = {
        "车牌号":"车牌号","车牌":"车牌号","plate":"车牌号",
        "是否可用":"是否可用","enabled":"是否可用",
        "核载数量":"核载数量","capacity_birds":"核载数量",
        "长度规格":"长度规格","length_m":"长度规格",
        "承包区域":"承包区域","region":"承包区域",
        "初始位置":"初始位置","home_company":"初始位置",
    }
    fl.rename(columns={k:v for k,v in mapping.items() if k in fl.columns}, inplace=True)
    need = ["车牌号","是否可用","核载数量","长度规格","承包区域"]
    miss = [c for c in need if c not in fl.columns]
    if miss: raise ValueError(f"fleet.csv 缺列：{miss}")
    fl["是否可用"] = pd.to_numeric(fl["是否可用"], errors="coerce").fillna(0).astype(int)
    fl["核载数量"] = pd.to_numeric(fl["核载数量"], errors="coerce").fillna(0).astype(int)
    fl["长度规格"] = pd.to_numeric(fl["长度规格"], errors="coerce").fillna(0.0).astype(float)
    fl["承包区域"] = fl["承包区域"].astype(str)
    fl["初始位置"] = fl.get("初始位置", pd.Series([""]*len(fl))).astype(str)
    fl = fl[fl["是否可用"]==1].copy()
    fl["承包区域_set"] = fl["承包区域"].apply(lambda s: [x.strip() for x in str(s).replace("，",",").split(",") if x.strip()])
    fl["区域ID"] = fl["承包区域_set"].apply(lambda xs: "|".join(sorted(xs)))
    return fl

def _cooling_days(time_h: float) -> int:
    # 当天可复用：若 2*时长<=12h，则返回 0
    if pd.isna(time_h): return 1
    rt = 2.0 * float(time_h)
    return 0 if rt <= 12.0 else int(math.ceil(rt/24.0))

def _build_od_maps(net: pd.DataFrame):
    dist, cool = {}, {}
    for _, r in net.iterrows():
        i, j = r["出发地"], r["到达地"]
        d = float(r["运输距离"])
        h = float(r["运输时长"]) if not pd.isna(r["运输时长"]) else float("nan")
        for a,b in [(i,j),(j,i)]:
            dist[(a,b)] = d
            cool[(a,b)] = _cooling_days(h)
    return dist, cool

# ---------- 主过程 ----------
def run_model(args):
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flow = _read_flow(out_dir)
    net = _read_network(data_dir)
    fleet = _read_fleet(data_dir)

    dist_map, cool_map = _build_od_maps(net)

    # 区域-城市关系
    fleet["class_id"] = fleet["车牌号"]
    cls_info = fleet.set_index("class_id").to_dict(orient="index")
    region2cities = {}
    for rid, grp in fleet.groupby("区域ID"):
        s = set()
        for xs in grp["承包区域_set"]:
            s.update(xs)
        region2cities[rid] = s
    city2regions = defaultdict(set)
    for rid, cs in region2cities.items():
        for c in cs: city2regions[c].add(rid)

    # ---- “需求桶”精确到（发车日,到达日,出发地,到达地,品种,性别） ----
    buckets = (flow.groupby(["日期","到达日期","出发地","到达地","品种","性别"], as_index=False)["数量"]
               .sum().rename(columns={"日期":"发车日"}))
    if buckets.empty: raise ValueError("流向方案_LP.csv 中没有有效记录")

    # ---- 车次候选：按同（发车日,到达日, 出发地,到达地）的总量，估计需要的车数 ----
    #     车次仍允许“拼车”，所以 x 会把不同(品种,性别)分量分配到同一车次
    pair_sum = (buckets.groupby(["发车日","到达日期","出发地","到达地"], as_index=False)["数量"].sum())
    min_cap = BRACKETS[0][1]  # 16000

    trips, bucket_rows = [], []
    tid = 0
    for _, r in pair_sum.iterrows():
        ship, arr = pd.to_datetime(r["发车日"]), pd.to_datetime(r["到达日期"])
        i, j = r["出发地"], r["到达地"]
        Q_total = float(r["数量"])
        miss = (i,j) not in dist_map
        # 记录所有明细桶（后面用于 x 的过滤）
        # 但车次只按 pair_sum 生成，避免重复
        if miss: continue
        dkm = dist_map[(i,j)]
        coold = cool_map.get((i,j), 1)
        # 需要的“车次槽位”
        N = max(1, int(math.ceil(Q_total / min_cap)))
        for _ in range(N):
            trips.append({
                "trip_id": tid, "发车日": ship, "到达日": arr,
                "i": i, "j": j, "dist_km": dkm, "cool_days": coold
            })
            tid += 1

    # 按明细桶建立 B；若无路网，则直接记未满足
    B = []
    unmet_if_noedge = []
    for _, r in buckets.iterrows():
        ship, arr = pd.to_datetime(r["发车日"]), pd.to_datetime(r["到达日期"])
        i, j = r["出发地"], r["到达地"]
        p, s, Q = r["品种"], r["性别"], float(r["数量"])
        miss = (i,j) not in dist_map
        if miss:
            unmet_if_noedge.append({
                "发车日": ship.strftime("%Y-%m-%d"),
                "到达日": arr.strftime("%Y-%m-%d"),
                "出发地": i, "到达地": j,
                "品种": p, "性别": s,
                "未满足数量": int(round(Q)),
                "未满足原因": "无可行路网"
            })
        else:
            B.append((len(B), ship, arr, i, j, p, s, Q))

    if not trips and not B:
        # 全是无路网
        pd.DataFrame(unmet_if_noedge).to_csv(out_dir/"未满足统计_派车阶段.csv",
                                            index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["发车日","到达日","出发地","到达地","车牌号","区域ID","品种","性别","数量"]).to_csv(
            out_dir/"派车计划.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"汇总日期":"TOTAL","总趟费元":0.0}]).to_csv(
            out_dir/"派车总成本汇总.csv", index=False, encoding="utf-8-sig")
        print("⚠️ 路网不可行，已输出全欠配。")
        return

    trips_df = pd.DataFrame(trips)

    # ====== 构建 MIP ======
    m = pl.LpProblem("DispatchPoolingCooling_FixedDates", pl.LpMinimize)

    TIDS = list(trips_df["trip_id"])  # 全部车次
    # 只为“该车可从该发车城市出发”的 (tid,cls) 建 z 变量
    z_keys = []
    for tid in TIDS:
        i_city = trips_df.loc[trips_df["trip_id"]==tid, "i"].item()
        for cls, info in cls_info.items():
            if i_city in region2cities[info["区域ID"]]:
                z_keys.append((tid, cls))

    # 变量
    # x: 每个需求桶 b 分到车次 tid 的数量（拼车）
    x = pl.LpVariable.dicts("x", ((b[0],tid) for b in B for tid in TIDS), lowBound=0)
    u = pl.LpVariable.dicts("u", (b[0] for b in B), lowBound=0)  # 欠配（逐桶，含品种性别）
    # L: 车次总装载（跨品种性别的总和）
    L = pl.LpVariable.dicts("L", (tid for tid in TIDS), lowBound=0)
    # bsel: 车次选择的计价档位（二进制）
    bsel = pl.LpVariable.dicts("bsel", ((tid,ri) for tid in TIDS for ri,_ in enumerate(BRACKETS)), cat="Binary")
    # z: 车次使用的具体车辆（二进制）
    z = pl.LpVariable.dicts("z", (key for key in z_keys), cat="Binary")

    # 区域库存（按天）：由于发/到日定死，这里只需要考虑“发车日”的出库与“发车日+cool”回库
    # 构建时间轴
    days = sorted(set(list(buckets["发车日"].unique()) + list(buckets["到达日期"].unique())))
    if not days: days = []
    # 为了处理回库，扩充 7 天缓冲
    if days:
        time_axis = list(pd.date_range(min(days), max(days)+pd.Timedelta(days=7), freq="D"))
    else:
        time_axis = []

    t_index = {t:i for i,t in enumerate(time_axis)}
    RID_LIST = list(region2cities.keys())
    CLS_LIST = list(cls_info.keys())
    A = pl.LpVariable.dicts("A", ((rid, cls, t_index[d]) for rid in RID_LIST for cls in CLS_LIST for d in time_axis), lowBound=0)
    U = pl.LpVariable.dicts("U", ((rid, cls, t_index[d]) for rid in RID_LIST for cls in CLS_LIST for d in time_axis), lowBound=0)
    V = pl.LpVariable.dicts("V", ((rid, cls, t_index[d]) for rid in RID_LIST for cls in CLS_LIST for d in time_axis), lowBound=0)

    # 目标：先最小欠配，再最小运费（用大权重）
    cost_terms = []
    for tid in TIDS:
        dkm = float(trips_df.loc[trips_df["trip_id"]==tid, "dist_km"].item())
        for ri, (_,_,price) in enumerate(BRACKETS):
            cost_terms.append(dkm * price * bsel[(tid,ri)])
    m += PENALTY_UNMET * pl.lpSum(u[b[0]] for b in B) + pl.lpSum(cost_terms)

    # 档位 & 装载范围
    for tid in TIDS:
        m += pl.lpSum(bsel[(tid,ri)] for ri,_ in enumerate(BRACKETS)) <= 1
        m += L[tid] <= pl.lpSum(BRACKETS[ri][1] * bsel[(tid,ri)] for ri,_ in enumerate(BRACKETS))
        m += L[tid] >= pl.lpSum(max(BRACKETS[ri][0], MIN_BATCH) * bsel[(tid,ri)] for ri,_ in enumerate(BRACKETS))

    # 车次-车辆唯一性
    for tid in TIDS:
        legal = [cls for (t,cls) in z_keys if t==tid]
        if legal:
            m += pl.lpSum(z[(tid,cls)] for cls in legal) == pl.lpSum(bsel[(tid,ri)] for ri,_ in enumerate(BRACKETS))
        else:
            m += pl.lpSum(bsel[(tid,ri)] for ri,_ in enumerate(BRACKETS)) == 0
            m += L[tid] == 0

    # 容量：L <= 所选车辆容量
    for tid in TIDS:
        legal = [cls for (t,cls) in z_keys if t==tid]
        if legal:
            m += L[tid] <= pl.lpSum(cls_info[cls]["核载数量"] * z[(tid,cls)] for cls in legal)

    # 需求守恒（逐桶，包含品种 & 性别；车次 L 为该车次所有 x 之和）
    for b_id, ship, arr, i, j, p, s, Q in B:
        ok_tids = list(trips_df[(trips_df["i"]==i) & (trips_df["j"]==j) &
                                (trips_df["发车日"]==ship) & (trips_df["到达日"]==arr)]["trip_id"])
        if ok_tids:
            m += pl.lpSum(x[(b_id,tid)] for tid in ok_tids) + u[b_id] == Q
            for tid in ok_tids:
                m += x[(b_id,tid)] <= L[tid]
        else:
            m += u[b_id] == Q

    # 车次总载重 = 该车次对应的所有桶分配之和
    for tid in TIDS:
        sub = []
        tr = trips_df.loc[trips_df["trip_id"]==tid].iloc[0]
        for b_id, ship, arr, i, j, p, s, Q in B:
            if i==tr["i"] and j==tr["j"] and ship==tr["发车日"] and arr==tr["到达日"]:
                sub.append(x[(b_id,tid)])
        if sub:
            m += L[tid] == pl.lpSum(sub)
        else:
            m += L[tid] == 0

    # 区域车池：库存 + 出车/回库（按发车日、发车日+cool）
    if time_axis:
        t0 = t_index[time_axis[0]]
        for rid in RID_LIST:
            for cls in CLS_LIST:
                init = 1 if cls_info[cls]["区域ID"]==rid else 0
                m += A[(rid,cls,t0)] == init

        # 逐日平衡
        for tpos, day in enumerate(time_axis[:-1]):
            next_pos = tpos + 1
            # 当天出车
            depart_terms = {(rid,cls):[] for rid in RID_LIST for cls in CLS_LIST}
            # 当天回库
            return_terms = {(rid,cls):[] for rid in RID_LIST for cls in CLS_LIST}

            # 收集出车
            for _, r in trips_df[trips_df["发车日"]==day].iterrows():
                tid = int(r["trip_id"]); i_city = r["i"]
                for rid in RID_LIST:
                    if i_city in region2cities[rid]:
                        for cls in CLS_LIST:
                            if (tid,cls) in z and cls_info[cls]["区域ID"]==rid:
                                depart_terms[(rid,cls)].append(z[(tid,cls)])

            # 收集回库
            for _, r in trips_df.iterrows():
                tid = int(r["trip_id"])
                ret_day = r["发车日"] + pd.Timedelta(days=int(r["cool_days"]))
                if ret_day == day:
                    for rid in RID_LIST:
                        for cls in CLS_LIST:
                            if (tid,cls) in z and cls_info[cls]["区域ID"]==rid:
                                return_terms[(rid,cls)].append(z[(tid,cls)])

            for rid in RID_LIST:
                for cls in CLS_LIST:
                    if cls_info[cls]["区域ID"] != rid:
                        m += A[(rid,cls,tpos)] == 0
                        m += U[(rid,cls,tpos)] == 0
                        m += V[(rid,cls,tpos)] == 0
                        m += A[(rid,cls,next_pos)] == 0
                        continue
                    # 出/回数量
                    if depart_terms[(rid,cls)]:
                        m += U[(rid,cls,tpos)] == pl.lpSum(depart_terms[(rid,cls)])
                    else:
                        m += U[(rid,cls,tpos)] == 0
                    if return_terms[(rid,cls)]:
                        m += V[(rid,cls,tpos)] == pl.lpSum(return_terms[(rid,cls)])
                    else:
                        m += V[(rid,cls,tpos)] == 0
                    # 当天可复用：出车 ≤ 库存 + 当天返还
                    m += U[(rid,cls,tpos)] <= A[(rid,cls,tpos)] + V[(rid,cls,tpos)]
                    # 库存流转
                    m += A[(rid,cls,next_pos)] == A[(rid,cls,tpos)] - U[(rid,cls,tpos)] + V[(rid,cls,tpos)]

    # ---------- 进度与规模信息 ----------
    n_x = sum(1 for _ in x)
    n_L = len(TIDS)
    n_bsel = len(TIDS) * len(BRACKETS)
    n_z = len(z_keys)
    n_AUV = (len(RID_LIST)*len(CLS_LIST)*len(time_axis)*3) if time_axis else 0
    print("====== 模型规模（预估） ======")
    print(f"需求桶数 B: {len(B)}")
    print(f"候选车次 TIDS: {len(TIDS)}")
    print(f"x 连续变量: ~{n_x}, L: {n_L}, bsel(二进制): {n_bsel}, z(二进制): {n_z}, A/U/V: ~{n_AUV}")
    print(f"时间轴: {len(time_axis)} 天，时间限制: {TIME_LIMIT_SEC if args.time_limit is None else args.time_limit} 秒")
    print("================================")

    log_path = out_dir / "cbc_log.txt"
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 开始求解，日志写入: {log_path}")
    solver = pl.PULP_CBC_CMD(
        msg=True,
        timeLimit=int(args.time_limit),
        logPath=str(log_path),
        options=["-stat=1"]
    )
    m.solve(solver)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 求解结束，状态: {pl.LpStatus[m.status]}")

    # ---------- 输出 ----------
    def sel_bracket(tid):
        for ri,_ in enumerate(BRACKETS):
            v = bsel[(tid,ri)].value()
            if v is not None and v > 0.5: return ri
        return None

    # 1) 派车计划：按车次×桶展开，输出【品种、性别、数量】，不含任何价格列
    rows = []
    z_choose = {}
    for (tid,cls) in z_keys:
        if z[(tid,cls)].value() and z[(tid,cls)].value() > 0.5:
            z_choose[tid] = cls
    for b_id, ship, arr, i, j, p, s, Q in B:
        ok_tids = list(trips_df[(trips_df["i"]==i)&(trips_df["j"]==j)&
                                (trips_df["发车日"]==ship)&(trips_df["到达日"]==arr)]["trip_id"])
        for tid in ok_tids:
            xv = x[(b_id,tid)].value()
            if xv is not None and xv > 1e-6:
                cls = z_choose.get(tid, "")
                rid = cls_info[cls]["区域ID"] if cls in cls_info else ""
                rows.append({
                    "发车日": ship.strftime("%Y-%m-%d"),
                    "到达日": arr.strftime("%Y-%m-%d"),
                    "出发地": i, "到达地": j,
                    "车牌号": cls, "区域ID": rid,
                    "品种": p, "性别": s,
                    "数量": int(round(xv))
                })
    dispatch_df = pd.DataFrame(rows).sort_values(["发车日","出发地","到达地","车牌号","品种","性别"])
    if dispatch_df.empty:
        dispatch_df = pd.DataFrame(columns=["发车日","到达日","出发地","到达地","车牌号","区域ID","品种","性别","数量"])
    dispatch_df.to_csv(out_dir/"派车计划.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 派车计划.csv 生成（{len(dispatch_df)} 行）")

    # 2) 派车总成本汇总：成本按“车次”计算一次，再按发车日汇总
    cost_rows = []
    for tid in TIDS:
        # 若车次未被激活则跳过
        if sum((bsel[(tid,ri)].value() or 0) for ri,_ in enumerate(BRACKETS)) < 0.5:
            continue
        tr = trips_df.loc[trips_df["trip_id"]==tid].iloc[0]
        dkm = float(tr["dist_km"])
        ri = sel_bracket(tid)
        if ri is None: continue
        unit = BRACKETS[ri][2]
        cost_rows.append({"发车日": tr["发车日"].strftime("%Y-%m-%d"),
                          "车次成本": unit * dkm})
    if cost_rows:
        cost_df = pd.DataFrame(cost_rows)
        day_sum = (cost_df.groupby("发车日", as_index=False)["车次成本"].sum()
                   .rename(columns={"车次成本":"总趟费元"}))
        total_row = pd.DataFrame([{"汇总日期":"TOTAL","总趟费元": round(day_sum["总趟费元"].sum(),2)}])
        total_cost_df = pd.concat([day_sum.rename(columns={"发车日":"汇总日期"}), total_row], ignore_index=True)
    else:
        total_cost_df = pd.DataFrame([{"汇总日期":"TOTAL","总趟费元": 0.0}])
    total_cost_df.to_csv(out_dir/"派车总成本汇总.csv", index=False, encoding="utf-8-sig")
    print("✅ 派车总成本汇总.csv 生成")

    # 3) 未满足统计（含品种、性别）
    unmet_rows = []
    for b_id, ship, arr, i, j, p, s, Q in B:
        ok_tids = list(trips_df[(trips_df["i"]==i)&(trips_df["j"]==j)&
                                (trips_df["发车日"]==ship)&(trips_df["到达日"]==arr)]["trip_id"])
        loaded = 0.0
        for tid in ok_tids:
            xv = x[(b_id,tid)].value()
            if xv is not None: loaded += float(xv)
        lack = Q - loaded
        if lack > 0.5:
            unmet_rows.append({
                "发车日": ship.strftime("%Y-%m-%d"),
                "到达日": arr.strftime("%Y-%m-%d"),
                "出发地": i, "到达地": j,
                "品种": p, "性别": s,
                "未满足数量": int(round(lack)),
                "未满足原因": "车辆/容量受限"
            })
    # 加上“无路网”的欠配
    unmet_rows.extend(unmet_if_noedge)

    # 3) 未满足统计（含品种、性别）
    unmet_rows = []
    for b_id, ship, arr, i, j, p, s, Q in B:
        ok_tids = list(trips_df[(trips_df["i"] == i) & (trips_df["j"] == j) &
                                (trips_df["发车日"] == ship) & (trips_df["到达日"] == arr)]["trip_id"])
        loaded = 0.0
        for tid in ok_tids:
            xv = x[(b_id, tid)].value()
            if xv is not None:
                loaded += float(xv)
        lack = Q - loaded
        if lack > 0.5:
            unmet_rows.append({
                "发车日": ship.strftime("%Y-%m-%d"),
                "到达日": arr.strftime("%Y-%m-%d"),
                "出发地": i, "到达地": j,
                "品种": p, "性别": s,
                "未满足数量": int(round(lack)),
                "未满足原因": "车辆/容量受限"
            })

    # 加上“无路网”的欠配
    unmet_rows.extend(unmet_if_noedge)

    # >>> 这里开始新增健壮性处理 <<<
    unmet_cols = ["发车日", "到达日", "出发地", "到达地", "品种", "性别", "未满足数量", "未满足原因"]
    if unmet_rows:
        unmet_df = pd.DataFrame(unmet_rows)[unmet_cols].sort_values(
            ["发车日", "到达日", "出发地", "到达地", "品种", "性别"]
        )
    else:
        unmet_df = pd.DataFrame(columns=unmet_cols)

    if not unmet_df.empty:
        unmet_df.to_csv(out_dir / "未满足统计_派车阶段.csv", index=False, encoding="utf-8-sig")
        print(f"⚠️ 未满足统计_派车阶段.csv 生成（{len(unmet_df)} 行）")
    else:
        print("✅ 派车阶段无欠配")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir",  type=str, default="./data_out")
    ap.add_argument("--time_limit", type=int, default=TIME_LIMIT_SEC)
    args = ap.parse_args()
    run_model(args)

if __name__ == "__main__":
    main()
