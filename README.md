# 隐私守护助手 (Privacy Guard)

基于 OpenCV 的实时摄像头人脸检测与自动切屏工具。当笔记本摄像头检测到陌生人、人脸数量变化或背景异常运动时，自动执行 `Win + D` 显示桌面，保护你的隐私。

## 功能特性

- **陌生人检测**：通过 LBPH 人脸识别模型区分「主人」与「陌生人」
- **人脸数量监控**：检测画面中人数的增加或减少
- **背景运动检测**：基于 MOG2 算法检测画面中的显著运动（即使未拍到正脸）
- **自动切屏**：满足触发条件时自动执行 Win+D 最小化所有窗口
- **冷却机制**：防止短时间内重复触发
- **日志记录**：所有触发事件和程序状态记录到日志文件

## 运行环境

| 依赖 | 最低版本 |
|------|----------|
| Python | 3.9+ |
| opencv-python | 4.8.0 |
| opencv-contrib-python | 4.8.0 |
| pyautogui | 0.9.54 |
| numpy | 1.24.0 |

## 快速开始

### 1. 安装依赖

```bash
pip install opencv-python opencv-contrib-python pyautogui numpy
```

### 2. 录入人脸

```bash
cd C:\Projects\privacy-guard
python enroll_face.py
```

- 面对摄像头，按 **SPACE** 拍摄照片（建议 10-20 张，覆盖不同角度）
- 按 **ESC** 完成拍摄并自动训练识别模型

### 3. 启动守护

```bash
python main.py
```

- 预览窗口中绿色框 = 主人，红色框 = 陌生人
- 按 **Q** 键退出程序

## 项目结构

```
privacy-guard/
├── main.py                # 主程序：实时检测 + 触发切屏
├── enroll_face.py         # 人脸录入与模型训练
├── config.py              # 配置文件（阈值、冷却时间等参数）
├── requirements.txt       # Python 依赖清单
├── README.md              # 项目说明
├── known_faces/           # 人脸数据（运行后生成）
│   ├── owner_000.jpg      # 参考人脸图片
│   └── trained_model.yml  # 训练好的 LBPH 模型
└── logs/                  # 日志文件
    └── privacy_guard.log
```

## 配置说明

编辑 `config.py` 可调整以下关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CAMERA_INDEX` | 0 | 摄像头设备索引，外接摄像头改为 1 |
| `HAAR_SCALE_FACTOR` | 1.1 | 人脸检测灵敏度，越小越敏感 |
| `FACE_RECOGNITION_THRESHOLD` | 70 | 人脸识别阈值，越低越严格 |
| `COOLDOWN_SECONDS` | 5 | 两次切屏最小间隔（秒） |
| `MOTION_RATIO_THRESHOLD` | 0.05 | 运动检测灵敏度（5% 画面变化） |
| `SHOW_PREVIEW` | True | 是否显示摄像头预览窗口 |

## 触发条件

| 条件 | 说明 |
|------|------|
| A — 陌生人 | 检测到非主人的人脸 |
| B — 人数变化 | 画面中人脸数量发生增减 |
| C — 背景运动 | 无人脸时画面出现显著运动 |

## 系统权限设置

### 摄像头权限
> Windows 设置 → 隐私和安全性 → 相机 → 开启「允许应用访问你的相机」

### 禁止休眠（运行期间）
```bash
powercfg -change -standby-timeout-ac 0
powercfg -change -monitor-timeout-ac 0
```

## 常见问题

| 问题 | 解决方法 |
|------|----------|
| 无法打开摄像头 | 检查 Windows 隐私设置中的相机权限，关闭占用摄像头的应用 |
| 检测不到人脸 | 调整 `config.py` 中的 `HAAR_SCALE_FACTOR` 为 1.05 |
| 切屏无反应 | 手动按 Win+D 确认快捷键正常；尝试以管理员身份运行终端 |
| 误触发频繁 | 增大 `COOLDOWN_SECONDS` 或提高 `FACE_RECOGNITION_THRESHOLD` |

## 技术原理

- **人脸检测**：OpenCV Haar Cascade（`haarcascade_frontalface_default.xml`）
- **人脸识别**：OpenCV LBPHFaceRecognizer（局部二值模式直方图）
- **运动检测**：OpenCV MOG2 背景减除算法
- **桌面控制**：pyautogui 模拟 Win+D 快捷键

## 许可证

MIT License
