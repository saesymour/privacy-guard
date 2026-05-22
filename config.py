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
# 邻近矩形最小数量 — 越大误检越少但漏检越多，推荐 4 ~ 6
HAAR_MIN_NEIGHBORS = 5
# 最小人脸尺寸（像素），小于此值的检测结果忽略
HAAR_MIN_FACE_SIZE = (40, 40)

# ==================== 人脸识别设置 ====================
# LBPH 人脸识别距离阈值 — 距离越小越相似，高于此值视为陌生人
FACE_RECOGNITION_THRESHOLD = 70.0

# ==================== 运动检测设置 ====================
# 运动区域占比阈值 — 画面中变化的像素比例超过此值时触发
MOTION_RATIO_THRESHOLD = 0.05  # 5%
# MOG2 背景减除器参数
MOTION_HISTORY = 500       # 历史帧数，用于学习背景
MOTION_VAR_THRESHOLD = 36  # 方差阈值，越低越敏感

# ==================== 触发策略设置 ====================
# 冷却时间（秒）— 两次触发操作之间的最小间隔，防止频繁切屏
COOLDOWN_SECONDS = 5
# 运动持续帧数 — 连续检测到运动的帧数达到此值时触发（防止误触发）
MOTION_CONSECUTIVE_FRAMES = 3
# 陌生人持续帧数 — 连续检测到陌生人的帧数达到此值时触发
STRANGER_CONSECUTIVE_FRAMES = 3

# ==================== 显示设置 ====================
# 是否显示摄像头预览窗口（调试用，正常运行建议开启）
SHOW_PREVIEW = True
# 预览窗口名称
WINDOW_NAME = "Privacy Guard - 隐私守护助手"

# ==================== 路径设置 ====================
KNOWN_FACES_DIR = "known_faces"       # 已知人脸图片存放目录
TRAINED_MODEL_PATH = "known_faces/trained_model.yml"  # 训练好的识别模型路径
LOG_DIR = "logs"                       # 日志目录
LOG_FILE = "logs/privacy_guard.log"    # 日志文件路径
