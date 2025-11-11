# scripts/run_all.py
import subprocess, os
from pathlib import Path
from datetime import datetime
from load_config import load_config
from utils import ensure_dir, save_yaml
from merge_to_excel import merge_csv_to_excel

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_ROOT = ROOT / "data_out"

# 1) 读取配置
cfg = load_config()
run_cfg = cfg.get("run", {})
p1 = cfg.get("phase1", {})
p2 = cfg.get("phase2", {})
p3 = cfg.get("phase3", {})

# 2) 创建带时间戳输出目录 + 保存配置快照
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = ensure_dir(OUT_ROOT / f"run_{timestamp}")
save_yaml(cfg, OUT_DIR / "config_used.yaml")
LOG_PATH = OUT_DIR / "run.log"

def run_and_tee(cmd: list[str], log_file: Path):
    """运行子进程，实时把 stdout/stderr 同时写到控制台和日志（强制 UTF-8）。"""
    print("\n>>> 运行：", " ".join(cmd))
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        # 强制子进程也启用 UTF-8（解决 Windows 控制台 GBK 问题）
        full_cmd = ["python", "-X", "utf8", *cmd[1:]] if cmd and cmd[0].endswith("python") else cmd
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",   # 父进程用 UTF-8 读取
            env=env,
        )
        for line in proc.stdout:
            print(line, end="")
            lf.write(line)
        ret = proc.wait()
        lf.write(f"\n[exit={ret}]\n")
        if ret != 0:
            raise subprocess.CalledProcessError(ret, full_cmd)

def main():
    commands = [
        [
            "python", str(ROOT / "core" / "phase1_local_match.py"),
            "--data_dir", str(DATA_DIR),
            "--out_dir", str(OUT_DIR),
            "--start_date", run_cfg["start_date"],
            "--end_date", run_cfg["end_date"],
            "--local_gap_ratio", str(p1["local_gap_ratio"]),
        ],
        [
            "python", str(ROOT / "core" / "phase2_od_assign.py"),
            "--data_dir", str(DATA_DIR),
            "--out_dir", str(OUT_DIR),
            "--start_date", run_cfg["start_date"],
            "--end_date", run_cfg["end_date"],
            "--rho", str(p2["rho"]),
            "--alpha", str(p2["alpha"]),
            "--age_penalty_le25", str(p2["age_penalty_le25"]),
            "--small_threshold", str(p2["small_threshold"]),
            "--vehicle_capacity", str(p2["vehicle_capacity"]),
            "--q_min_hint", str(p2["q_min_hint"]),
        ],
        [
            "python", str(ROOT / "core" / "phase3_dispatch.py"),
            "--data_dir", str(DATA_DIR),
            "--out_dir", str(OUT_DIR),
            "--time_limit", str(p3["time_limit"]),
        ],
    ]

    try:
        for i, cmd in enumerate(commands, 1):
            print(f"\n========== 阶段 {i} ==========")
            run_and_tee(cmd, LOG_PATH)

        print(f"\n✅ 全部完成，输出目录：{OUT_DIR}")
        print(f"📝 运行日志：{LOG_PATH}")

        # 自动合并 CSV → Excel 到当次 run 目录
        print("\n📊 自动合并所有 CSV → Excel...")
        output_excel = f"run_{timestamp}.xlsx"
        merge_csv_to_excel(OUT_DIR, output_excel)
        print("✅ Excel 汇总完成！")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 阶段命令失败：{' '.join(e.cmd)} (exit={e.returncode})")
        print(f"请查看日志：{LOG_PATH}")
        raise

if __name__ == "__main__":
    main()
