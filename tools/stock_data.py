import akshare as ak
import pandas as pd
import time
from datetime import datetime, timedelta
from langchain_core.tools import tool
from loguru import logger


def _today() -> str:
    return datetime.today().strftime("%Y%m%d")

def _days_ago(n: int) -> str:
    return (datetime.today() - timedelta(days=n)).strftime("%Y%m%d")

def _normalize_code(ts_code: str) -> str:
    """600519.SH → 600519"""
    return ts_code.split(".")[0]

def _sina_code(ts_code: str) -> str:
    """600519.SH → sh600519，000001.SZ → sz000001"""
    parts = ts_code.upper().split(".")
    code = parts[0]
    exchange = parts[1] if len(parts) > 1 else ("sh" if code.startswith("6") else "sz")
    return exchange.lower() + code

def _retry(fn, *args, retries=3, delay=2, **kwargs):
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i < retries - 1:
                logger.debug(f"重试 {i+1}/{retries-1}: {e}")
                time.sleep(delay)
            else:
                raise


@tool
def get_stock_basic_info(ts_code: str) -> str:
    """获取股票基础信息：公司名称、行业、上市日期等。ts_code 格式如 600519.SH"""
    code = _normalize_code(ts_code)

    # 方法1: 东方财富个股信息
    try:
        df = _retry(ak.stock_individual_info_em, symbol=code)
        info = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        return (
            f"股票代码: {ts_code}\n"
            f"公司名称: {info.get('股票简称', code)}\n"
            f"所属行业: {info.get('行业', 'N/A')}\n"
            f"上市日期: {info.get('上市时间', 'N/A')}\n"
            f"总股本: {info.get('总股本', 'N/A')}\n"
            f"流通股: {info.get('流通股', 'N/A')}\n"
            f"总市值: {info.get('总市值', 'N/A')}\n"
            f"流通市值: {info.get('流通市值', 'N/A')}"
        )
    except Exception as e1:
        logger.warning(f"个股信息方法1失败: {e1}")

    # 方法2: 从新浪行情获取股票名称 + THS财务确认存在
    try:
        df_price = _retry(ak.stock_zh_a_daily, symbol=_sina_code(ts_code), adjust="qfq")
        latest_price = float(df_price.iloc[-1]["close"])
        # THS 财务摘要确认股票存在并获取财务年度范围
        fin = _retry(ak.stock_financial_abstract_ths, symbol=code, indicator="按年度")
        year_range = f"{fin.iloc[0]['报告期']}—{fin.iloc[-1]['报告期']}" if not fin.empty else "N/A"
        return (
            f"股票代码: {ts_code}\n"
            f"最新价: {latest_price:.2f}\n"
            f"财务数据: {year_range} 共{len(fin)}期\n"
            f"（公司名称获取失败，建议通过 --name 参数指定）"
        )
    except Exception as e2:
        logger.warning(f"个股信息方法2失败: {e2}")

    return f"股票代码: {ts_code}\n公司名称: {code}"


@tool
def get_stock_price(ts_code: str, days: int = 60) -> str:
    """获取股票近 N 天的日线行情数据。ts_code 格式如 600519.SH"""
    code = _normalize_code(ts_code)

    # 方法1: 新浪行情（稳定）
    try:
        df = _retry(ak.stock_zh_a_daily, symbol=_sina_code(ts_code), adjust="qfq")
        df = df.tail(days).reset_index(drop=True)
        if df.empty:
            raise ValueError("empty")
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        pct = (float(latest["close"]) - float(prev["close"])) / float(prev["close"]) * 100
        return (
            f"最新交易日: {latest['date']}\n"
            f"收盘价: {float(latest['close']):.2f}  涨跌幅: {pct:+.2f}%\n"
            f"最高: {float(latest['high']):.2f}  最低: {float(latest['low']):.2f}\n"
            f"成交量: {float(latest['volume']):.0f}手\n"
            f"近{days}日共{len(df)}条，期间最高: {df['high'].astype(float).max():.2f}，"
            f"最低: {df['low'].astype(float).min():.2f}"
        )
    except Exception as e1:
        logger.warning(f"新浪行情失败，尝试东方财富: {e1}")

    # 方法2: 东方财富（备用）
    try:
        df = _retry(
            ak.stock_zh_a_hist, symbol=code, period="daily",
            start_date=_days_ago(days), end_date=_today(), adjust="qfq",
        )
        if df.empty:
            return f"未获取到 {ts_code} 行情数据"
        df = df.sort_values("日期")
        latest = df.iloc[-1]
        pct = float(latest["涨跌幅"])
        return (
            f"最新交易日: {latest['日期']}\n"
            f"收盘价: {latest['收盘']:.2f}  涨跌幅: {pct:+.2f}%\n"
            f"最高: {latest['最高']:.2f}  最低: {latest['最低']:.2f}\n"
            f"成交量: {latest['成交量']:.0f}手\n"
            f"近{days}日共{len(df)}条"
        )
    except Exception as e2:
        logger.error(f"get_stock_price error: {e2}")
        return f"获取行情失败: {str(e2)}"


@tool
def get_financial_indicators(ts_code: str) -> str:
    """获取股票财务指标：净利润、营收增速、PE、PB 等。ts_code 格式如 600519.SH"""
    code = _normalize_code(ts_code)

    # 同花顺财务摘要（取最新2期，从新到旧）
    fin_text = "财务摘要暂无"
    try:
        fin = _retry(ak.stock_financial_abstract_ths, symbol=code, indicator="按年度")
        if not fin.empty:
            recent = fin.iloc[::-1].head(2)  # 反转后取最新2期
            rows = []
            for _, row in recent.iterrows():
                items = {k: v for k, v in row.items() if pd.notna(v) and str(v).strip()}
                row_str = "  ".join(f"{k}: {v}" for k, v in items.items())
                rows.append(row_str)
            fin_text = "\n".join(rows)
    except Exception as e:
        logger.warning(f"同花顺财务数据失败: {e}")

    # 实时 PE/PB（东方财富，失败不影响主流程）
    valuation_text = "PE/PB 暂无"
    try:
        spot = _retry(ak.stock_zh_a_spot_em, retries=2, delay=1)
        row = spot[spot["代码"] == code]
        if not row.empty:
            r = row.iloc[0]
            valuation_text = (
                f"PE(动态): {r.get('市盈率-动态', 'N/A')}  "
                f"PB: {r.get('市净率', 'N/A')}  "
                f"总市值: {r.get('总市值', 'N/A')}"
            )
    except Exception:
        pass

    return f"=== 年度财务摘要（最新2期）===\n{fin_text}\n\n=== 实时估值 ===\n{valuation_text}"


@tool
def get_price_dataframe(ts_code: str, days: int = 120) -> str:
    """获取股票历史行情 CSV，用于沙盒画图。ts_code 格式如 600519.SH"""
    code = _normalize_code(ts_code)

    # 方法1: 新浪行情
    try:
        df = _retry(ak.stock_zh_a_daily, symbol=_sina_code(ts_code), adjust="qfq")
        df = df.tail(days).reset_index(drop=True)
        df = df.rename(columns={
            "date": "trade_date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "vol",
        })
        return df[["trade_date", "open", "high", "low", "close", "vol"]].to_csv(index=False)
    except Exception as e1:
        logger.warning(f"新浪 CSV 失败，尝试东方财富: {e1}")

    # 方法2: 东方财富备用
    try:
        df = _retry(
            ak.stock_zh_a_hist, symbol=code, period="daily",
            start_date=_days_ago(days), end_date=_today(), adjust="qfq",
        )
        df = df.sort_values("日期").rename(columns={
            "日期": "trade_date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "vol",
        })
        return df[["trade_date", "open", "high", "low", "close", "vol"]].to_csv(index=False)
    except Exception as e2:
        logger.error(f"get_price_dataframe error: {e2}")
        return f"ERROR: {str(e2)}"
