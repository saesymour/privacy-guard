"""
隐私守护助手 — 主程序
======================
功能：实时检测笔记本摄像头画面中的异常情况，自动执行 Win+D 切屏。
触发条件：
  A. 摄像头前出现非本人的陌生面孔
  B. 画面中人脸数量发生变化（有人进入或离开）
  C. 背景出现显著运动（即使未检测到正脸）

所有操作均记录到日志文件。

使用前请先运行 enroll_face.py 录入人脸。
"""

import cv2
import os
import sys
import time
import logging
import numpy as np
import pyautogui

# 导入配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

# pyautogui 安全设置 — 鼠标移到左上角可紧急中止
pyautogui.FAILSAFE = True


# ============================================================
# 日志系统初始化
# ============================================================
def setup_logging():
    """配置日志：同时输出到文件和控制台"""
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    logger = logging.getLogger("PrivacyGuard")
    logger.setLevel(logging.INFO)

    # 文件处理器 — 详细日志
    fh = logging.FileHandler(cfg.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # 控制台处理器 — 简要输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = setup_logging()


# ============================================================
# 人脸检测器（基于 OpenCV Haar Cascade，内置离线上）
# ============================================================
class FaceDetector:
    """使用 OpenCV Haar Cascade 检测画面中的人脸，返回人脸边界框列表"""

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"无法加载人脸检测模型: {cascade_path}")
        logger.info("人脸检测器已就绪 (Haar Cascade)")

    def detect(self, frame_bgr):
        """
        检测一帧画面中的人脸
        参数: frame_bgr — BGR格式的numpy数组
        返回: [(x, y, w, h), ...] 人脸边界框列表
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=cfg.HAAR_SCALE_FACTOR,
            minNeighbors=cfg.HAAR_MIN_NEIGHBORS,
            minSize=cfg.HAAR_MIN_FACE_SIZE,
        )
        return [tuple(f) for f in faces]

    def release(self):
        pass  # Haar Cascade 不需要显式释放


# ============================================================
# 人脸识别器（基于 OpenCV LBPH）
# ============================================================
class FaceRecognizer:
    """加载训练好的 LBPH 模型，判断检测到的人脸是否为「主人」"""

    def __init__(self):
        self.recognizer = None
        self.enrolled = False
        if os.path.exists(cfg.TRAINED_MODEL_PATH):
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(cfg.TRAINED_MODEL_PATH)
                self.enrolled = True
                logger.info(f"已加载人脸模型: {cfg.TRAINED_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"加载人脸模型失败: {e}")
                logger.warning("将工作在「仅检测人脸」模式（无法区分主人和陌生人）")
        else:
            logger.warning("未找到训练好的人脸模型，请先运行 enroll_face.py")
            logger.warning("将工作在「仅检测人脸」模式（无法区分主人和陌生人）")

    def predict(self, face_gray):
        """
        预测人脸归属
        参数: face_gray — 灰度人脸图像 (200x200)
        返回: (is_owner, distance) — is_owner为True表示主人，distance为LBPH距离
        """
        if not self.enrolled:
            return None, None  # 无法判断

        label, distance = self.recognizer.predict(face_gray)
        is_owner = distance < cfg.FACE_RECOGNITION_THRESHOLD
        return is_owner, distance


# ============================================================
# 运动检测器（基于 OpenCV MOG2）
# ============================================================
class MotionDetector:
    """使用背景减除算法检测画面中的运动区域"""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.MOTION_HISTORY,
            varThreshold=cfg.MOTION_VAR_THRESHOLD,
            detectShadows=False,
        )
        self.motion_consecutive = 0

    def detect(self, frame_bgr):
        """
        检测运动程度
        参数: frame_bgr — BGR格式的numpy数组
        返回: (has_motion, motion_ratio)
        """
        fg_mask = self.bg_subtractor.apply(frame_bgr)
        # 计算前景像素占比
        total_pixels = fg_mask.size
        motion_pixels = np.count_nonzero(fg_mask)
        motion_ratio = motion_pixels / total_pixels

        has_motion = motion_ratio > cfg.MOTION_RATIO_THRESHOLD

        # 连续帧计数（过滤短暂抖动）
        if has_motion:
            self.motion_consecutive += 1
        else:
            self.motion_consecutive = 0

        return self.motion_consecutive >= cfg.MOTION_CONSECUTIVE_FRAMES, motion_ratio


# ============================================================
# 桌面控制器
# ============================================================
class DesktopController:
    """执行系统级操作：切屏"""

    @staticmethod
    def show_desktop():
        """执行 Win+D 显示桌面"""
        try:
            pyautogui.hotkey("win", "d")
            logger.info(">>> 已执行 Win+D 显示桌面 <<<")
            return True
        except Exception as e:
            logger.error(f"执行 Win+D 失败: {e}")
            return False


# ============================================================
# 隐私守护主控
# ============================================================
class PrivacyGuard:
    """主控制器，协调检测→判断→触发→执行的全流程"""

    def __init__(self):
        # 检测模块
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.motion_detector = MotionDetector()
        self.desktop = DesktopController()

        # 状态追踪
        self.last_trigger_time = 0           # 上次触发时间戳（用于冷却）
        self.last_face_count = 0             # 上一帧的人脸数量
        self.face_count_stable = 0           # 稳定的人脸数量
        self.face_count_stable_counter = 0   # 人脸数量稳定帧计数
        self.stranger_consecutive = 0        # 连续检测到陌生人的帧数

        # 陌生人触发标志（防止同一事件重复触发）
        self.stranger_already_triggered = False

        logger.info("=" * 50)
        logger.info("隐私守护助手已启动")
        logger.info(f"人脸识别状态: {'已就绪' if self.face_recognizer.enrolled else '仅检测模式（请先录入人脸）'}")
        logger.info(f"触发冷却: {cfg.COOLDOWN_SECONDS} 秒")
        logger.info("按 Q 键退出程序")
        logger.info("=" * 50)

    def run(self):
        """主循环"""
        cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)

        if not cap.isOpened():
            logger.error("无法打开摄像头！请检查 Windows 隐私设置中的摄像头权限")
            sys.exit(1)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("摄像头读帧失败，重试中...")
                    time.sleep(0.1)
                    continue

                # 水平翻转（镜像效果）
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()

                # ---- 1. 人脸检测 ----
                faces = self.face_detector.detect(frame)
                face_count = len(faces)
                strangers_detected = False

                # ---- 2. 人脸识别（如果模型已训练） ----
                owner_count = 0
                stranger_count = 0
                for (x, y, w, h) in faces:
                    # 裁剪人脸区域并转为灰度
                    face_roi = frame[y:y + h, x:x + w]
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.resize(face_gray, (200, 200))

                    is_owner, distance = self.face_recognizer.predict(face_gray)

                    if is_owner is True:
                        owner_count += 1
                        color = (0, 255, 0)  # 绿色 = 主人
                        label = "Owner"
                    elif is_owner is False:
                        stranger_count += 1
                        color = (0, 0, 255)  # 红色 = 陌生人
                        label = f"Stranger ({distance:.0f})"
                    else:
                        # 模型不可用，仅显示检测到的人脸
                        color = (255, 255, 0)  # 黄色 = 未知
                        label = "Unknown"
                        stranger_count += 1  # 未知人脸视为潜在陌生人

                    # 画框和标签
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(display_frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if stranger_count > 0 or (not self.face_recognizer.enrolled and face_count > 0):
                    strangers_detected = True

                # ---- 3. 运动检测 ----
                has_motion, motion_ratio = self.motion_detector.detect(frame)

                # ---- 4. 人脸数量变化检测 ----
                # 使用稳定计数器避免短时波动（检测闪烁）
                if face_count == self.last_face_count:
                    self.face_count_stable_counter += 1
                else:
                    self.face_count_stable_counter = 0

                face_count_changed = False
                if self.face_count_stable_counter >= 5:  # 连续5帧稳定后确认
                    if self.face_count_stable != face_count:
                        if self.face_count_stable != 0 or face_count != 0:
                            face_count_changed = True
                            logger.info(f"人脸数量变化: {self.face_count_stable} -> {face_count}")
                        self.face_count_stable = face_count
                    self.face_count_stable_counter = 0

                # ---- 5. 陌生人连续帧计数 ----
                if strangers_detected:
                    self.stranger_consecutive += 1
                else:
                    self.stranger_consecutive = 0
                    self.stranger_already_triggered = False  # 离开后重置

                # ---- 6. 判断触发条件 ----
                trigger_reason = None
                now = time.time()

                # 冷却时间内不触发
                if now - self.last_trigger_time < cfg.COOLDOWN_SECONDS:
                    pass
                else:
                    # 条件 A：陌生人出现
                    if (self.stranger_consecutive >= cfg.STRANGER_CONSECUTIVE_FRAMES
                            and not self.stranger_already_triggered):
                        trigger_reason = "陌生人出现在摄像头前"
                        self.stranger_already_triggered = True

                    # 条件 B：人脸数量变化
                    elif face_count_changed and face_count > 0:
                        trigger_reason = f"人脸数量变化: {self.face_count_stable} -> {face_count}"

                    # 条件 C：显著运动（无人脸时）
                    elif has_motion and face_count == 0:
                        trigger_reason = f"检测到背景显著运动 ({motion_ratio:.1%})"

                # ---- 7. 执行触发 ----
                if trigger_reason:
                    logger.warning(f"触发！原因: {trigger_reason}")
                    self.desktop.show_desktop()
                    self.last_trigger_time = now

                # ---- 8. 更新状态 ----
                self.last_face_count = face_count

                # ---- 9. 显示预览 ----
                cv2.putText(display_frame, f"Faces: {face_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, f"Motion: {motion_ratio:.1%}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if has_motion else (0, 255, 0), 2)

                # 状态栏 — 底部
                cooldown_left = cfg.COOLDOWN_SECONDS - (now - self.last_trigger_time)
                if cooldown_left > 0:
                    status = f"COOLDOWN: {cooldown_left:.1f}s"
                    status_color = (0, 165, 255)
                elif strangers_detected:
                    status = "ALERT: Stranger!"
                    status_color = (0, 0, 255)
                else:
                    status = "SAFE"
                    status_color = (0, 255, 0)

                cv2.putText(display_frame, status, (10, display_frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(display_frame, "Q=Quit", (10, display_frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 冷却进度条
                if cooldown_left > 0:
                    bar_width = int((cooldown_left / cfg.COOLDOWN_SECONDS) * 200)
                    cv2.rectangle(display_frame, (10, 455), (10 + bar_width, 465),
                                  (0, 165, 255), -1)

                if cfg.SHOW_PREVIEW:
                    cv2.imshow(cfg.WINDOW_NAME, display_frame)

                # ---- 10. 按键处理 ----
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    logger.info("用户手动退出程序")
                    break

        except KeyboardInterrupt:
            logger.info("收到中断信号，正在退出...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_detector.release()
            logger.info("隐私守护助手已停止")


# ============================================================
# 入口
# ============================================================
def main():
    guard = PrivacyGuard()
    guard.run()


if __name__ == "__main__":
    main()
