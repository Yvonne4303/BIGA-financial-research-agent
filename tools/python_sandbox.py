from langchain_core.tools import tool
from config.settings import E2B_API_KEY, CHARTS_DIR
from loguru import logger
import base64
import os
import io
import sys
import traceback

CHART_SETUP = """
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import mplfinance as mpf
import warnings
warnings.filterwarnings('ignore')
"""


def _run_local(code: str, stock_csv: str) -> str:
    """本地执行 Python 代码（E2B 不可用时的 fallback）"""
    import pandas as pd
    namespace: dict = {}

    if stock_csv and not stock_csv.startswith("ERROR"):
        df_raw = pd.read_csv(io.StringIO(stock_csv))
        namespace["df_raw"] = df_raw

    os.makedirs(CHARTS_DIR, exist_ok=True)
    # 让代码里的 savefig 把图存到 CHARTS_DIR
    save_path = os.path.join(CHARTS_DIR, "chart_0.png")
    full_code = CHART_SETUP + "\n"
    full_code += f"_CHART_PATH = {repr(save_path)}\n"
    full_code += code.replace("fname='chart.png'", f"fname={repr(save_path)}")

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    try:
        exec(full_code, namespace)
        output = captured.getvalue()
    except Exception:
        output = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    result_parts = []
    if output:
        result_parts.append(f"输出:\n{output}")
    if os.path.exists(save_path):
        result_parts.append(f"图表已保存: {save_path}")
    return "\n".join(result_parts) if result_parts else "代码执行完成，无输出"


@tool
def run_python_code(code: str, stock_csv: str = "") -> str:
    """
    执行 Python 代码用于技术指标计算和图表生成。
    优先使用 E2B 云沙盒，无 key 时自动切换本地执行。
    stock_csv: CSV 格式行情数据，变量名为 df_raw。
    """
    if not E2B_API_KEY:
        logger.info("E2B_API_KEY 未配置，使用本地模式执行")
        return _run_local(code, stock_csv)

    try:
        from e2b_code_interpreter import Sandbox
        with Sandbox(api_key=E2B_API_KEY) as sbx:
            if stock_csv and not stock_csv.startswith("ERROR"):
                inject = f"""
import pandas as pd
import io
_csv_data = '''{stock_csv}'''
df_raw = pd.read_csv(io.StringIO(_csv_data))
"""
                sbx.run_code(inject)

            result = sbx.run_code(CHART_SETUP + "\n" + code)

            output_parts = []
            if result.logs.stdout:
                output_parts.append("输出:\n" + "\n".join(result.logs.stdout))
            if result.logs.stderr:
                output_parts.append("错误:\n" + "\n".join(result.logs.stderr))

            saved = []
            for i, artifact in enumerate(result.results):
                if hasattr(artifact, "png") and artifact.png:
                    os.makedirs(CHARTS_DIR, exist_ok=True)
                    path = os.path.join(CHARTS_DIR, f"chart_{i}.png")
                    with open(path, "wb") as f:
                        f.write(base64.b64decode(artifact.png))
                    saved.append(path)

            if saved:
                output_parts.append(f"图表已保存: {', '.join(saved)}")

            return "\n".join(output_parts) if output_parts else "代码执行完成，无输出"
    except Exception as e:
        logger.warning(f"E2B 执行失败，切换本地模式: {e}")
        return _run_local(code, stock_csv)


# 预置的图表生成代码模板
KLINE_CODE_TEMPLATE = """
df = df_raw.copy()
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.set_index('trade_date')
df.index.name = 'Date'
df = df.rename(columns={{'open':'Open','high':'High','low':'Low','close':'Close','vol':'Volume'}})
df = df[['Open','High','Low','Close','Volume']].astype(float)

# 计算 MACD
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['Hist'] = df['MACD'] - df['Signal']

# 计算 MA
df['MA5'] = df['Close'].rolling(5).mean()
df['MA20'] = df['Close'].rolling(20).mean()

df_plot = df.tail(60).copy()
add_plots = [
    mpf.make_addplot(df_plot['MA5'], color='orange', width=1),
    mpf.make_addplot(df_plot['MA20'], color='blue', width=1),
    mpf.make_addplot(df_plot['MACD'], panel=2, color='purple', width=1),
    mpf.make_addplot(df_plot['Signal'], panel=2, color='red', width=1),
    mpf.make_addplot(df_plot['Hist'], panel=2, type='bar', color='grey'),
]

mpf.plot(df_plot, type='candle', style='charles',
         addplot=add_plots, volume=True,
         title='{title}',
         figsize=(14, 10),
         savefig=dict(fname='chart.png', dpi=150))
print("K线图+MACD绘制完成")
"""
