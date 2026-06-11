"""
智能日志分析器 — 基于 MiMo Chat API
====================================
读取隐私守护助手的运行日志，调用 MiMo LLM 生成日报分析：
- 触发事件统计与趋势
- 误报识别与参数优化建议
- 异常时间段分布
- 安全风险摘要

可手动运行或设为定时任务（cron / Task Scheduler）。

API 格式：OpenAI 兼容（base_url: https://api.xiaomimimo.com/v1）
模型：mimo-v2.5-pro（支持 100 万 token 上下文，适合长日志分析）
"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Optional

from openai import OpenAI

logger = logging.getLogger("PrivacyGuard.DailyReporter")


class DailyReporter:
    """日志分析 + 日报生成"""

    DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
    DEFAULT_MODEL = "mimo-v2.5-pro"

    SYSTEM_PROMPT = """\
You are a security log analyst for a laptop privacy protection tool called "Privacy Guard".
The tool monitors the laptop webcam and triggers desktop hiding (Win+D) when it detects:
- Stranger faces (people not recognized as the owner)
- Changes in face count (someone entered or left)
- Significant background motion (when no faces are present)

Analyze the provided log data and produce a daily report in JSON:

{
  "summary": {
    "total_triggers": <number>,
    "by_reason": { "stranger": <n>, "face_count_change": <n>, "motion": <n> },
    "peak_hour": "<time range>",
    "false_positive_estimate": "<low|medium|high>"
  },
  "anomalies": [
    { "time": "<timestamp>", "description": "<unusual pattern>" }
  ],
  "recommendations": [
    "<specific config adjustment or usage tip>"
  ],
  "risk_assessment": "<one paragraph summary of privacy risk level>"
}

If the log is empty (no triggers), set risk_assessment to nothing recorded."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("MIMO_API_KEY", "")
        if not api_key:
            raise ValueError("未设置 MIMO_API_KEY，请设置环境变量或传入 api_key")

        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    @staticmethod
    def _empty_report(recommendations=None, risk="No data."):
        return {
            "summary": {"total_triggers": 0, "by_reason": {}, "peak_hour": "N/A",
                        "false_positive_estimate": "N/A"},
            "anomalies": [],
            "recommendations": recommendations or [],
            "risk_assessment": risk,
        }

    def read_log(self, log_path: str, max_lines: int = 500) -> str:
        """读取日志文件内容，限制最大行数"""
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"日志文件不存在: {log_path}")

        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 取最近的 max_lines 行（日志可能很大）
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
            truncated = True
        else:
            truncated = False

        content = "".join(lines)
        if truncated:
            content = f"(log truncated, showing last {max_lines} lines)\n\n" + content

        return content

    def analyze(self, log_path: str, date: Optional[str] = None) -> dict:
        """
        分析日志文件并生成日报

        参数:
            log_path: 日志文件路径
            date: 日期字符串（YYYY-MM-DD），默认今天

        返回: 解析后的日报 dict
        """
        date = date or datetime.now().strftime("%Y-%m-%d")

        # 读取日志
        try:
            log_content = self.read_log(log_path)
        except FileNotFoundError:
            return self._empty_report(recommendations=["日志文件不存在，请先运行主程序生成日志"])

        prompt = f"""Date: {date}
Tool: Privacy Guard v1.0
Log entries:

{log_content}

---

Generate a daily report based on the log above."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.2,
                timeout=30.0,
            )
        except Exception as e:
            logger.error(f"API 调用失败: {type(e).__name__}")
            return self._empty_report(recommendations=[f"API 调用失败: {type(e).__name__}"], risk="Report generation failed.")

        raw_content = response.choices[0].message.content
        if not raw_content:
            logger.warning("API 返回空 content")
            return self._empty_report(recommendations=["API 返回空响应"], risk="Report generation failed.")
        raw = raw_content.strip()

        # 提取 JSON（处理 ```json ... ``` 格式）
        if '`' in raw:
            m = re.search(r"`(?:json)?\s*\n(.*?)\n\s*`", raw, re.DOTALL)
            if m:
                raw = m.group(1).strip()

        try:
            report = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("JSON 解析失败，使用原始文本作为风险评估")
            report = self._empty_report(risk=raw[:500])

        tokens = response.usage.total_tokens if response.usage else 0
        report["_meta"] = {
            "tokens_used": tokens,
            "model": self.model,
            "date": date,
        }

        return report

    def print_report(self, report: dict):
        """格式化打印日报"""
        meta = report.get("_meta", {})
        summary = report.get("summary", {})

        print("=" * 60)
        print(f"  Privacy Guard — 日报分析")
        print(f"  日期: {meta.get('date', 'N/A')}")
        print(f"  Token 消耗: {meta.get('tokens_used', 0)}")
        print("=" * 60)
        print()
        print(f"  总计触发次数: {summary.get('total_triggers', 0)}")
        print(f"  触发原因分布: {summary.get('by_reason', {})}")
        print(f"  高峰时段:     {summary.get('peak_hour', 'N/A')}")
        print(f"  误报评估:     {summary.get('false_positive_estimate', 'N/A')}")
        print()

        anomalies = report.get("anomalies", [])
        if anomalies:
            print("  异常事件:")
            for a in anomalies:
                print(f"    [{a.get('time', '?')}] {a.get('description', '')}")
            print()

        recs = report.get("recommendations", [])
        if recs:
            print("  优化建议:")
            for r in recs:
                print(f"    • {r}")
            print()

        risk = report.get("risk_assessment", "")
        if risk:
            print(f"  风险评估: {risk}")
            print()

        print("=" * 60)


# ============================================================
# 命令行入口
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Privacy Guard 智能日志分析 — 基于 MiMo LLM 生成日报"
    )
    parser.add_argument(
        "log_path", nargs="?",
        default="logs/privacy_guard.log",
        help="日志文件路径 (默认: logs/privacy_guard.log)",
    )
    parser.add_argument(
        "--date", default=None,
        help="分析日期 (YYYY-MM-DD)，默认为今天",
    )
    parser.add_argument(
        "--json", action="store_true", dest="output_json",
        help="以 JSON 格式输出",
    )
    args = parser.parse_args()

    try:
        reporter = DailyReporter()
        report = reporter.analyze(args.log_path, args.date)

        if args.output_json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            reporter.print_report(report)

    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        print("请设置环境变量: set MIMO_API_KEY=sk-xxxxx", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
