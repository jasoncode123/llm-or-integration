# 🚚 LLM–OR Integration: 智能调拨优化系统

结合大语言模型（LLM）与运筹优化（OR）的运输调拨与派车系统。  
通过一键执行脚本完成 **三阶段调拨优化**，并自动输出汇总报表。

---

## 📁 项目结构
core/ # 三个阶段算法（本地自配 / OD分配 / 派车调度）
scripts/ # 一键运行脚本、配置加载、Excel合并
data/ # 输入数据
data_out/ # 输出结果（自动生成 run_时间戳 目录）
config.yaml # 参数配置文件
README.md # 项目说明文件

yaml
复制代码

---

## ⚙️ 快速开始

### 1️⃣ 激活虚拟环境
```bash
conda activate llm_or
2️⃣ 一键运行
bash
复制代码
python scripts/run_all.py
运行后会自动生成：

data_out/run_YYYYMMDD_HHMMSS/ 目录

包含阶段输出、日志 run.log、配置快照 config_used.yaml

自动汇总文件 合并汇总_YYYYMMDD.xlsx

🧩 主要依赖
库名	用途
pandas / numpy	数据处理
PuLP / OR-Tools	运筹优化求解
pyyaml	读取配置文件
xlsxwriter	Excel 报表合并

安装依赖
bash
复制代码
pip install pandas numpy pulp ortools pyyaml xlsxwriter
🧱 功能亮点
✅ 一键运行全流程
✅ 参数配置化（config.yaml）
✅ 自动按时间戳归档输出
✅ 自动生成 Excel 汇总
✅ 运行日志与配置留痕

🧰 常见问题
Q1: Windows 控制台报 UnicodeEncodeError: 'gbk'
👉 解决方法：

bash
复制代码
chcp 65001
set PYTHONUTF8=1
Q2: 输出文件没合并？
👉 检查 data_out/ 是否存在或有权限。脚本会自动创建。

📌 作者信息
Maintainer: sunrise

Email: 27246712@qq.com

Date: 2025-11-11

🧾 .gitignore
bash
复制代码
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/

# IDE & Virtual Env
.venv/
.env
.idea/
.vscode/

# OS cache
.DS_Store
Thumbs.db

# Outputs & Logs
data_out/
logs/

# Excel 临时文件
~$*.xlsx
⚙️ config.yaml（推荐结构）
yaml
复制代码
# ========================
# LLM–OR 调拨系统配置文件
# ========================

data_dir: "./data"
out_dir: "./data_out"

start_date: "2025-07-01"
end_date: "2025-07-02"

# 阶段 1：本地匹配
local_gap_ratio: 0.05

# 阶段 2：流向分配
rho: 0.000176
alpha: 1.0
age_penalty_le25: 0.05
small_threshold: 200
vehicle_capacity: 16000
q_min_hint: 8000

# 阶段 3：派车调度
time_limit: 600