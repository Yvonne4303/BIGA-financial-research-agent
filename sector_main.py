#!/usr/bin/env python3
"""
A股行业板块轮动分析系统
用法:
  python sector_main.py --period 日    # 今日板块涨跌分析
  python sector_main.py --period 周    # 本周板块涨跌分析
  python sector_main.py --period 月    # 本月板块涨跌分析
"""
import os
import requests
from urllib.parse import urlparse

# 绕过系统代理，直连国内数据源
_DIRECT_DOMAINS = {
    "eastmoney.com", "sina.com.cn", "10jqka.com.cn",
    "finance.qq.com", "gtimg.com", "sinajs.cn",
    "finance.sina.com.cn", "ths.com.cn", "xueqiu.com",
}
_orig_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    host = urlparse(request.url).hostname or ""
    if any(host == d or host.endswith("." + d) for d in _DIRECT_DOMAINS):
        kwargs["proxies"] = {"http": None, "https": None}
    return _orig_send(self, request, **kwargs)
requests.Session.send = _patched_send

import argparse
from loguru import logger
from workflow.sector_graph import run_sector_analysis


def main():
    parser = argparse.ArgumentParser(description="A股行业板块轮动分析")
    parser.add_argument(
        "--period", default="日",
        choices=["日", "周", "月"],
        help="分析周期：日（今日）/ 周（本周）/ 月（本月）",
    )
    parser.add_argument(
        "--user", default="default_user",
        help="用户ID，用于读取个人持仓/偏好记忆",
    )
    args = parser.parse_args()

    period_label = {"日": "今日", "周": "本周", "月": "本月"}[args.period]
    logger.info(f"开始{period_label}行业板块分析")

    print(f"\n{'='*60}")
    print(f"  A股行业板块轮动分析 — {period_label}")
    print(f"{'='*60}\n")

    report = run_sector_analysis(period=args.period, user_id=args.user)

    print("\n" + "="*60)
    print(f"{period_label}板块轮动报告")
    print("="*60)
    print(report)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
