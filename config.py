"""
隐私守护助手 - 配置文件
所有可调参数集中管理，修改后重启程序即可生效。
"""

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
MIMO_TRIGGER_CONDITIONS = ["stranger", "face_change"]

# ==================== 路径设置 ====================
KNOWN_FACES_DIR = "known_faces"       # 已知人脸图片存放目录
TRAINED_MODEL_PATH = "known_faces/trained_model.yml"  # 训练好的识别模型路径
LOG_DIR = "logs"                       # 日志目录
LOG_FILE = "logs/privacy_guard.log"    # 日志文件路径
