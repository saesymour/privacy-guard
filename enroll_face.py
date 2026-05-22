"""
人脸录入脚本 — 拍摄并注册「主人」的面部信息
==============================================
用途：运行此脚本，对着摄像头拍摄多张面部照片，
      训练人脸识别模型，作为后续陌生人检测的基准。

使用流程：
  1. 确保只有你一个人在摄像头前
  2. 运行此脚本：python enroll_face.py
  3. 按 SPACE 键拍摄（建议拍 10-20 张，覆盖不同角度和表情）
  4. 按 ESC 键完成拍摄并自动训练模型
"""

import cv2
import os
import sys
import numpy as np

# 导入配置
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def main():
    print("=" * 60)
    print("  隐私守护助手 — 人脸录入程序")
    print("=" * 60)
    print()
    print("  操作说明：")
    print("    SPACE 键 — 拍摄一张照片")
    print("    ESC 键  — 完成拍摄，开始训练模型")
    print("    Q 键    — 直接退出（不训练）")
    print()

    # 确保目录存在
    os.makedirs(cfg.KNOWN_FACES_DIR, exist_ok=True)

    # 初始化摄像头
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头，请检查：")
        print("  1. 摄像头是否被其他程序占用")
        print("  2. Windows 隐私设置中是否允许应用访问摄像头")
        sys.exit(1)

    # 初始化 OpenCV Haar Cascade 人脸检测器（OpenCV 内置，无需下载）
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print(f"[错误] 无法加载人脸检测模型: {cascade_path}")
        sys.exit(1)

    captured_count = 0
    face_images = []  # 存储检测到的人脸灰度图
    face_labels = []  # 标签（本项目中只有一个人，标签均为0）

    print("[信息] 摄像头已就绪，请开始拍摄...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[警告] 读取摄像头画面失败")
            continue

        # 水平翻转（镜像效果，更自然）
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # 转灰度图用于检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用 Haar Cascade 检测人脸
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=cfg.HAAR_SCALE_FACTOR,
            minNeighbors=cfg.HAAR_MIN_NEIGHBORS,
            minSize=cfg.HAAR_MIN_FACE_SIZE,
        )

        faces_found = len(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 显示状态信息
        cv2.putText(display_frame, f"Faces: {faces_found}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Captured: {captured_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "SPACE=Capture | ESC=Finish | Q=Quit", (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if cfg.SHOW_PREVIEW:
            cv2.imshow(cfg.WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE 键 — 拍摄
            if len(faces) == 1:
                # 只有画面中正好有一张脸时才拍摄
                (x, y, w, h) = faces[0]
                if w > 0 and h > 0:
                    face_roi = frame[y:y + h, x:x + w]
                    face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    # 统一缩放到 200x200
                    face_gray = cv2.resize(face_gray, (200, 200))
                    face_images.append(face_gray)
                    face_labels.append(0)  # 标签 0 = 主人
                    captured_count += 1
                    print(f"  [拍摄] 第 {captured_count} 张照片已保存")
            elif len(faces) == 0:
                print("  [警告] 未检测到人脸，请面对摄像头")
            else:
                print("  [警告] 检测到多张人脸！请确保只有你一个人在画面中")

        elif key == 27:  # ESC 键 — 完成拍摄，开始训练
            break
        elif key == ord('q') or key == ord('Q'):
            print("\n[信息] 用户取消录入")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    # ========== 训练 LBPH 人脸识别模型 ==========
    if len(face_images) < 5:
        print(f"\n[错误] 只拍摄了 {len(face_images)} 张照片，至少需要 5 张才能训练模型")
        print("  请重新运行此脚本并多拍几张照片")
        sys.exit(1)

    print(f"\n[训练] 共收集 {len(face_images)} 张人脸照片，开始训练模型...")

    # LBPH 识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )

    recognizer.train(face_images, np.array(face_labels))
    recognizer.write(cfg.TRAINED_MODEL_PATH)

    # 同时保存参考人脸图片（用于后续备用）
    for i, face_img in enumerate(face_images):
        cv2.imwrite(os.path.join(cfg.KNOWN_FACES_DIR, f"owner_{i:03d}.jpg"), face_img)

    print(f"[完成] 模型已保存到: {cfg.TRAINED_MODEL_PATH}")
    print(f"[完成] 参考人脸图片已保存到: {cfg.KNOWN_FACES_DIR}/")
    print(f"\n现在可以运行主程序了: python main.py")


if __name__ == "__main__":
    main()
