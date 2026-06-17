import akshare as ak
import pandas as pd
import time
from datetime import datetime, timedelta
from langchain_core.tools import tool
from tools.web_search import search_financial_news
from loguru import logger


def _retry(fn, *args, retries=3, delay=2, **kwargs):
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise


# 每个大板块对应多个同花顺细分子板块，取均值避免单一子板块偏差
SECTOR_THS_MAP: dict[str, list[str]] = {
    "食品饮料":  ["白酒", "饮料制造", "食品加工制造"],
    "医药生物":  ["化学制药", "生物制品", "中药", "医药商业"],
    "医疗器械":  ["医疗器械", "医疗服务"],
    "银行":      ["银行"],
    "非银金融":  ["证券", "保险", "多元金融"],
    "房地产":    ["房地产"],
    "半导体":    ["半导体", "元件", "光学光电子"],
    "消费电子":  ["消费电子", "其他电子", "黑色家电", "白色家电"],
    "计算机":    ["软件开发", "IT服务", "计算机设备"],
    "通信":      ["通信设备", "通信服务"],
    "有色金属":  ["工业金属", "贵金属", "小金属", "能源金属"],
    "国防军工":  ["军工装备", "军工电子"],
    "汽车":      ["汽车整车", "汽车零部件", "汽车服务及其他"],
    "新能源":    ["光伏设备", "风电设备", "电池", "其他电源设备"],
    "传媒娱乐":  ["文化传媒", "影视院线", "游戏"],
    "农林牧渔":  ["养殖业", "种植业与林业", "农产品加工", "农化制品"],
    "煤炭":      ["煤炭开采加工"],
    "钢铁":      ["钢铁"],
    "基础化工":  ["化学制品", "化学原料", "化学纤维", "塑料制品"],
    "建筑":      ["建筑材料", "建筑装饰"],
    "公用事业":  ["电力", "燃气", "环境治理"],
    "交通运输":  ["港口航运", "公路铁路运输", "物流"],
}


def _ths_period_return(ths_name: str, period: str) -> float | None:
    """用同花顺官方板块指数计算涨跌幅"""
    days_map = {"日": 5, "周": 10, "月": 40}
    lookback = days_map.get(period, 5)
    start = (datetime.today() - timedelta(days=lookback)).strftime("%Y%m%d")
    end   = datetime.today().strftime("%Y%m%d")
    try:
        df = _retry(ak.stock_board_industry_index_ths,
                    symbol=ths_name, start_date=start, end_date=end,
                    retries=2, delay=2)
        if df is None or df.empty or len(df) < 2:
            return None
        df = df.sort_values("日期")
        c_latest = float(df.iloc[-1]["收盘价"])
        if period == "日":
            c_base = float(df.iloc[-2]["收盘价"])
        else:
            c_base = float(df.iloc[0]["收盘价"])
        return round((c_latest - c_base) / c_base * 100, 2)
    except Exception as e:
        logger.debug(f"  THS [{ths_name}] 失败: {e}")
        return None


def _sector_avg_return(ths_names: list[str], period: str) -> float | None:
    """多个子板块涨跌幅取均值"""
    returns = [_ths_period_return(name, period) for name in ths_names]
    valid = [r for r in returns if r is not None]
    return round(sum(valid) / len(valid), 2) if valid else None


def get_all_sectors_performance(period: str = "日") -> list[dict]:
    """
    用同花顺官方板块指数计算各行业涨跌幅（多子板块均值）。
    period: "日" / "周" / "月"
    返回按涨跌幅排序的板块列表。
    """
    results = []
    total = len(SECTOR_THS_MAP)
    for i, (sector_name, ths_names) in enumerate(SECTOR_THS_MAP.items()):
        logger.debug(f"[{i+1}/{total}] {sector_name}（子板块:{ths_names}）")
        r = _sector_avg_return(ths_names, period)
        if r is not None:
            results.append({
                "name": sector_name,
                "ths_names": ths_names,
                "change_pct": r,
                "data_ok": True,
            })
        else:
            results.append({
                "name": sector_name,
                "ths_names": ths_names,
                "change_pct": 0.0,
                "data_ok": False,
            })

    valid = [r for r in results if r["data_ok"]]
    logger.info(f"成功获取 {len(valid)}/{total} 个板块数据（同花顺官方指数，多子板块均值）")
    return sorted(results, key=lambda x: x["change_pct"], reverse=True)


def get_sector_history(sector_name: str, days: int = 20) -> str:
    """获取板块近期历史走势（同花顺官方指数，取第一个子板块）"""
    ths_names = SECTOR_THS_MAP.get(sector_name, [sector_name])
    ths_name = ths_names[0] if isinstance(ths_names, list) else ths_names
    start = (datetime.today() - timedelta(days=days + 10)).strftime("%Y%m%d")
    end   = datetime.today().strftime("%Y%m%d")
    try:
        df = _retry(ak.stock_board_industry_index_ths,
                    symbol=ths_name, start_date=start, end_date=end)
        df = df.sort_values("日期").tail(days)
        latest = df.iloc[-1]
        first  = df.iloc[0]
        total_chg = (float(latest["收盘价"]) - float(first["收盘价"])) / float(first["收盘价"]) * 100
        return (
            f"同花顺[{ths_name}]近{len(df)}日: {total_chg:+.2f}%\n"
            f"最新收盘: {latest['收盘价']:.2f}  "
            f"期间最高: {df['最高价'].astype(float).max():.2f}  "
            f"最低: {df['最低价'].astype(float).min():.2f}"
        )
    except Exception as e:
        return f"历史数据获取失败: {e}"


@tool
def search_sector_news(sector_name: str, days: int = 7) -> str:
    """搜索行业板块的最新新闻、政策和资金动向。sector_name 如 '新能源' '医药生物'"""
    query = f"{sector_name}板块 行情 政策 资金 近期"
    return search_financial_news.invoke({"query": query, "max_results": 6})
