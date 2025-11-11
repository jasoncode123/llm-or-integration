# scripts/merge_to_excel.py
from pathlib import Path
import pandas as pd
import glob
import os

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data_out"

def merge_csv_to_excel(run_dir: Path, output_name):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / output_name

    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for file in glob.glob(str(run_dir / "*.csv")):
            sheet_name = os.path.splitext(os.path.basename(file))[0][:31]
            try:
                try:
                    df = pd.read_csv(file, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(file, encoding="gbk")
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"✅ 已写入 sheet：{sheet_name}")
            except Exception as e:
                print(f"⚠️ 处理文件 {file} 出错：{e}")

    print(f"\n📂 已生成：{output_path}")
