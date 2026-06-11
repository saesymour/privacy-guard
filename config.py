"""
隐私守护助手 - 配置文件
所有可调参数集中管理，修改后重启程序即可生效。
"""
import os

# ==================== 摄像头设置 ====================
CAMERA_INDEX = 0          # 摄像头设备索引，0 表示默认笔记本内置摄像头
CAMERA_WIDTH = 640        # 画面宽度（像素）
CAMERA_HEIGHT = 480       # 画面高度（像素）

# ==================== 人脸检测设置（OpenCV Haar Cascade） ====================
# 检测尺度缩放系数 — 越小越敏感但也越慢，推荐 1.05 ~ 1.2
HAAR_SCALE_FACTOR = 1.1
# 邻近矩形最小数量 — 越大误检越少但漏检越多，推荐 5 ~ 7
HAAR_MIN_NEIGHBORS = 6
# 最小人脸尺寸（像素），小于此值的检测结果忽略
HAAR_MIN_FACE_SIZE = (50, 50)

# ==================== 人脸识别设置 ====================
# LBPH 人脸识别距离阈值 — 距离越小越相似，高于此值视为陌生人
# 70=标准 / 80=宽松（允许侧脸等角度变化）/ 60=严格（只接受正面标准照）
FACE_RECOGNITION_THRESHOLD = 80.0

# ==================== 运动检测设置 ====================
# 运动区域占比阈值 — 画面中变化的像素比例超过此值时触发
MOTION_RATIO_THRESHOLD = 0.08  # 8%（原5%，提高以减少小幅身体晃动误触发）
# MOG2 背景减除器参数
MOTION_HISTORY = 800       # 历史帧数（原500，更长的背景学习降低噪声敏感度）
MOTION_VAR_THRESHOLD = 48  # 方差阈值（原36，更高则更难被小幅运动触发）

# ==================== 触发策略设置 ====================
# 冷却时间（秒）— 两次触发操作之间的最小间隔，防止频繁切屏
COOLDOWN_SECONDS = 15
# 运动持续帧数 — 连续检测到运动的帧数达到此值时触发（防止误触发）
MOTION_CONSECUTIVE_FRAMES = 5
# 陌生人持续帧数 — 连续检测到陌生人的帧数达到此值时触发
STRANGER_CONSECUTIVE_FRAMES = 6
# 人脸数量稳定帧数 — 连续检测到相同人脸数量多少帧后才确认变化（防止检测闪烁）
FACE_STABLE_FRAMES = 10
# 纯运动触发前需确认人脸已消失多少帧（防止人脸检测闪烁导致的连锁触发）
MOTION_FACE_ABSENCE_FRAMES = 30
# 纯运动触发的冷却倍数 — 运动触发后冷却时间 = COOLDOWN_SECONDS × 此值
MOTION_TRIGGER_COOLDOWN_MULTIPLIER = 3.0

# ==================== 显示设置 ====================
# 是否显示摄像头预览窗口（调试用，正常运行建议开启）
SHOW_PREVIEW = True
# 预览窗口名称
WINDOW_NAME = "Privacy Guard - 隐私守护助手"

# ==================== MiMo API 智能分析设置 ====================
# 是否启用 MiMo 智能场景分析（需要有效的 API Key）
SMART_ANALYZER_ENABLED = False
# MiMo API 密钥 — 从环境变量读取，也可直接填写（勿提交到 Git！）
MIMO_API_KEY = None  # 建议: os.getenv("MIMO_API_KEY")
# MiMo API 地址（OpenAI 兼容格式）
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
# 使用模型：mimo-v2.5（多模态）/ mimo-v2.5-pro（旗舰）/ mimo-v2-omni（全模态）
MIMO_MODEL = "mimo-v2.5"
# API 超时时间（秒）
MIMO_TIMEOUT = 5.0
# API 失败重试次数
MIMO_MAX_RETRIES = 2
# 最小分析间隔（秒）— 两次 API 调用之间至少间隔多久，控制 token 消耗
MIMO_MIN_INTERVAL = 2.0
# 哪些情况触发智能分析：stranger（陌生人出现）/ face_change（人数变化）/ motion（背景运动）
MIMO_TRIGGER_CONDITIONS = ["stranger", "face_change", "motion"]

# ==================== 路径设置 ====================
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(_PROJECT_ROOT, "known_faces")
TRAINED_MODEL_PATH = os.path.join(_PROJECT_ROOT, "known_faces", "trained_model.yml")
LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(_PROJECT_ROOT, "logs", "privacy_guard.log")

# ==================== 人脸识别训练参数 ====================
FACE_RESIZE_DIM = (200, 200)            # 人脸统一缩放尺寸
MIN_TRAINING_PHOTOS = 5                 # 最少拍摄照片数（含增强后的总数）
LBPH_RADIUS = 1
LBPH_NEIGHBORS = 8
LBPH_GRID_X = 8
LBPH_GRID_Y = 8

# ==================== 数据增强参数 ====================
AUG_BRIGHT_ALPHA = 1.3
AUG_BRIGHT_BETA = 20
AUG_DARK_ALPHA = 0.7
AUG_DARK_BETA = -20

# ==================== 日志设置 ====================
LOG_MAX_BYTES = 5 * 1024 * 1024         # 单个日志文件最大 5MB
LOG_BACKUP_COUNT = 3                    # 保留 3 个备份

# ==================== 摄像头容错设置 ====================
CAM_FAIL_THRESHOLD = 50                 # 连续失败多少次后自动退出
CAM_RETRY_SLEEP = 0.1                   # 读帧失败后等待秒数


def validate():
    """校验配置参数的合法性，在 import 时调用"""
    errors = []
    if not (0 < MOTION_RATIO_THRESHOLD < 1):
        errors.append(f"MOTION_RATIO_THRESHOLD 必须在 (0,1)，当前: {MOTION_RATIO_THRESHOLD}")
    if HAAR_SCALE_FACTOR <= 1.0:
        errors.append(f"HAAR_SCALE_FACTOR 必须 > 1.0，当前: {HAAR_SCALE_FACTOR}")
    if COOLDOWN_SECONDS <= 0:
        errors.append(f"COOLDOWN_SECONDS 必须 > 0，当前: {COOLDOWN_SECONDS}")
    if FACE_RECOGNITION_THRESHOLD < 0:
        errors.append(f"FACE_RECOGNITION_THRESHOLD 必须 >= 0，当前: {FACE_RECOGNITION_THRESHOLD}")
    if not isinstance(CAMERA_INDEX, (int, str)):
        errors.append(f"CAMERA_INDEX 必须是 int 或 str，当前: {type(CAMERA_INDEX).__name__}")
    if errors:
        raise ValueError("配置校验失败:\n  " + "\n  ".join(errors))


validate()
