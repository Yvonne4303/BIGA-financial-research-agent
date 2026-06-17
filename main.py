#!/usr/bin/env python3
"""
基于 LangGraph 的多智能体金融投研系统（大A）
用法: python main.py --code 600519.SH --user my_user_id
"""
import os
import requests
from urllib.parse import urlparse

# macOS 系统代理比 NO_PROXY 环境变量优先级更高，需要在 requests 层面强制直连
# 针对国内金融数据源（akshare 使用 requests，这里打补丁让其绕过代理）
_DIRECT_DOMAINS = {
    "eastmoney.com", "sina.com.cn", "10jqka.com.cn",
    "finance.qq.com", "gtimg.com", "akshare.xyz",
    "ths.com.cn", "xueqiu.com", "163.com",
    "sinajs.cn", "hq.sinajs.cn", "finance.sina.com.cn",
}

_orig_send = requests.Session.send

def _patched_send(self, request, **kwargs):
    host = urlparse(request.url).hostname or ""
    if any(host == d or host.endswith("." + d) for d in _DIRECT_DOMAINS):
        kwargs["proxies"] = {"http": None, "https": None}
    return _orig_send(self, request, **kwargs)

requests.Session.send = _patched_send

import argparse
import sys
from loguru import logger
from workflow.graph import run_research


def main():
    parser = argparse.ArgumentParser(description="A股多智能体投研系统")
    parser.add_argument("--code", required=True, help="股票代码，如 600519.SH")
    parser.add_argument("--user", default="default_user", help="用户ID（用于长期记忆）")
    parser.add_argument("--name", default="", help="公司名称（可选，不填自动获取）")
    args = parser.parse_args()

    logger.info(f"开始分析: {args.code} | 用户: {args.user}")
    print(f"\n{'='*60}")
    print(f"  多智能体投研系统 — {args.code}")
    print(f"{'='*60}\n")

    report = run_research(
        ts_code=args.code,
        user_id=args.user,
        company_name=args.name,
    )

    print("\n" + "="*60)
    print("最终投研报告")
    print("="*60)
    print(report)
    print("="*60 + "\n")
    return report


if __name__ == "__main__":
    main()
