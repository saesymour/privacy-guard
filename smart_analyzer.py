"""
智能场景分析器 — 基于 MiMo Vision API
======================================
使用 MiMo-V2.5 多模态模型对摄像头画面进行语义分析，
区分"同事路过""陌生人偷看""正常环境变化"等场景，
替代简单的规则触发，减少误报并提升隐私保护的智能化程度。

API 格式：OpenAI 兼容（base_url: https://api.xiaomimimo.com/v1）
模型：mimo-v2.5（多模态）或 mimo-v2.5-pro（旗舰推理）

使用前请设置环境变量 MIMO_API_KEY，或传入 api_key 参数。
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from openai import OpenAI

logger = logging.getLogger("PrivacyGuard.SmartAnalyzer")


@dataclass
class ThreatAnalysis:
    """威胁分析结果"""
    threat_level: str       # "low" | "medium" | "high"
    action: str             # "ignore" | "warn" | "hide"
    reason: str             # 判断理由（自然语言）
    scene_description: str  # 画面描述
    tokens_used: int        # 本次调用消耗 token 数
    latency_ms: float       # API 响应延迟（毫秒）


class SmartAnalyzer:
    """使用 MiMo Vision API 进行智能场景威胁分析"""

    # 默认配置
    DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
    DEFAULT_MODEL = "mimo-v2.5"

    # 分析提示词模板
    SYSTEM_PROMPT = """\
You are a privacy threat analyzer for a laptop screen privacy protection system.
Your job is to analyze the camera image and determine if there is a privacy threat.

Rules for threat assessment:
- LOW: Person walking by naturally, chatting with someone else, or far away. No threat.
- MEDIUM: Person standing nearby but not actively looking at screen. Voice warning suggested.
- HIGH: Person standing close behind, looking directly at the screen, or taking photos.
       Immediate desktop hiding required.

Key indicators:
- Eye gaze direction relative to the camera/screen
- Distance from the camera (close = higher threat)
- Body orientation (facing camera vs. facing away)
- Whether holding a phone/camera pointed at screen

Reply ONLY with valid JSON, no other text:
{
  "threat_level": "low|medium|high",
  "action": "ignore|warn|hide",
  "reason": "one sentence explaining your judgment",
  "scene_description": "what you see in the image"
}"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        enabled: bool = True,
        timeout: float = 5.0,
        max_retries: int = 2,
    ):
        """
        参数:
            api_key: MiMo API key，默认从环境变量 MIMO_API_KEY 读取
            base_url: API 地址，默认 https://api.xiaomimimo.com/v1
            model: 模型名称，默认 mimo-v2.5
            enabled: 是否启用智能分析（关闭时返回空结果）
            timeout: API 调用超时时间（秒）
            max_retries: 失败重试次数
        """
        self.enabled = enabled
        self.timeout = timeout
        self.max_retries = max_retries
        self.total_tokens_used = 0
        self.total_calls = 0
        self.total_latency_ms = 0.0

        if not enabled:
            logger.info("智能分析器未启用（enabled=False）")
            return

        api_key = api_key or os.getenv("MIMO_API_KEY", "")
        if not api_key:
            logger.warning("未设置 MIMO_API_KEY，智能分析功能不可用")
            logger.warning("请设置环境变量: set MIMO_API_KEY=sk-xxxxx")
            self.enabled = False
            return

        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self.DEFAULT_MODEL

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        logger.info(f"智能分析器已就绪 (model={self.model}, base_url={self.base_url})")

    def _encode_frame(self, frame_bgr: np.ndarray, quality: int = 60) -> str:
        """将 OpenCV BGR 帧编码为 base64 JPEG 字符串"""
        _, buffer = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode("utf-8")

    def _build_context_text(
        self,
        face_count: int,
        stranger_count: int,
        motion_ratio: float,
        has_motion: bool,
        cooldown_active: bool,
    ) -> str:
        """构建场景上下文文本，附加到图片提示中"""
        parts = []
        if face_count > 0:
            parts.append(f"{face_count} face(s) detected in frame")
            if stranger_count > 0:
                parts.append(f"{stranger_count} stranger(s) (not recognized as owner)")
        else:
            parts.append("No faces detected")

        if has_motion:
            parts.append(f"significant background motion ({motion_ratio:.1%} of frame)")
        else:
            parts.append("no significant motion")

        if cooldown_active:
            parts.append("(cooldown period active from previous trigger)")

        return "; ".join(parts)

    def analyze(
        self,
        frame_bgr: np.ndarray,
        face_count: int = 0,
        stranger_count: int = 0,
        motion_ratio: float = 0.0,
        has_motion: bool = False,
        cooldown_active: bool = False,
    ) -> Optional[ThreatAnalysis]:
        """
        分析摄像头画面中的隐私威胁

        返回 ThreatAnalysis 或 None（API 调用失败或未启用时）
        """
        if not self.enabled:
            return None

        t_start = time.time()

        # 编码帧
        try:
            image_base64 = self._encode_frame(frame_bgr)
        except Exception as e:
            logger.error(f"帧编码失败: {e}")
            return None

        context_text = self._build_context_text(
            face_count, stranger_count, motion_ratio, has_motion, cooldown_active
        )

        user_message = f"Scene context: {context_text}\n\nAnalyze the image above for privacy threats."

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ]

        # 带重试的 API 调用
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.1,  # 低温度确保输出一致性
                    timeout=self.timeout,
                )
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = 0.5 * (attempt + 1)
                    logger.warning(f"API 调用失败 (attempt {attempt + 1}), {wait}s 后重试: {e}")
                    time.sleep(wait)
        else:
            logger.error(f"API 调用失败，已用尽 {self.max_retries + 1} 次尝试: {last_error}")
            return None

        latency_ms = (time.time() - t_start) * 1000

        # 解析响应
        raw_content = response.choices[0].message.content.strip()
        usage = response.usage

        try:
            # 尝试提取 JSON（处理可能的 markdown 代码块包装）
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                raw_content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else raw_content

            result = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.warning(f"JSON 解析失败，原始响应: {raw_content[:200]}")
            # 降级：基于规则判断
            result = self._fallback_analysis(face_count, stranger_count, has_motion)

        # 规范化字段
        threat_level = result.get("threat_level", "low").lower()
        if threat_level not in ("low", "medium", "high"):
            threat_level = "low"

        action = result.get("action", "ignore").lower()
        if action not in ("ignore", "warn", "hide"):
            action = self._default_action(threat_level)

        tokens_used = usage.total_tokens if usage else 0
        self.total_tokens_used += tokens_used
        self.total_calls += 1
        self.total_latency_ms += latency_ms

        analysis = ThreatAnalysis(
            threat_level=threat_level,
            action=action,
            reason=result.get("reason", "No reason provided"),
            scene_description=result.get("scene_description", ""),
            tokens_used=tokens_used,
            latency_ms=round(latency_ms, 1),
        )

        level_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        logger.info(
            f"{level_emoji.get(threat_level, '?')} [{threat_level.upper()}] "
            f"action={action} | {analysis.reason} | "
            f"tokens={tokens_used} latency={latency_ms:.0f}ms"
        )

        return analysis

    def _fallback_analysis(self, face_count: int, stranger_count: int, has_motion: bool) -> dict:
        """当 API 响应无法解析时的降级规则判断"""
        if stranger_count > 0:
            return {
                "threat_level": "medium",
                "action": "warn",
                "reason": "Stranger detected (rule-based fallback due to API parse error)",
                "scene_description": "",
            }
        elif has_motion and face_count == 0:
            return {
                "threat_level": "low",
                "action": "ignore",
                "reason": "Motion without faces (rule-based fallback)",
                "scene_description": "",
            }
        return {
            "threat_level": "low",
            "action": "ignore",
            "reason": "No threat indicators found",
            "scene_description": "",
        }

    @staticmethod
    def _default_action(threat_level: str) -> str:
        return {"low": "ignore", "medium": "warn", "high": "hide"}.get(threat_level, "ignore")

    @property
    def stats(self) -> dict:
        """返回使用统计"""
        if self.total_calls == 0:
            return {"calls": 0, "total_tokens": 0, "avg_latency_ms": 0}
        return {
            "calls": self.total_calls,
            "total_tokens": self.total_tokens_used,
            "avg_latency_ms": round(self.total_latency_ms / self.total_calls, 1),
        }


# ============================================================
# 便捷函数：从配置创建 SmartAnalyzer
# ============================================================
def create_analyzer_from_config(config_module) -> SmartAnalyzer:
    """从 config.py 读取配置并创建 SmartAnalyzer 实例"""
    return SmartAnalyzer(
        api_key=getattr(config_module, "MIMO_API_KEY", None) or os.getenv("MIMO_API_KEY"),
        base_url=getattr(config_module, "MIMO_BASE_URL", None),
        model=getattr(config_module, "MIMO_MODEL", None),
        enabled=getattr(config_module, "SMART_ANALYZER_ENABLED", True),
        timeout=getattr(config_module, "MIMO_TIMEOUT", 5.0),
        max_retries=getattr(config_module, "MIMO_MAX_RETRIES", 2),
    )
