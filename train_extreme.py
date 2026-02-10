from ultralytics import YOLO
import os

# 确保权重文件已存在，避免重复下载
weight_path = "yolov8n.pt"
if not os.path.exists(weight_path):
    print("📥 正在下载yolov8n.pt权重文件...")
    model = YOLO(weight_path)  # 自动下载
else:
    print("✅ 本地已存在权重文件，直接加载")
    model = YOLO(weight_path)

# 极端测试：单张图片训练500轮
results = model.train(
    data='./project/data.yaml',       # 你的配置文件路径
    epochs=500,               # 大幅增加训练轮数，强制模型记住单张图
    imgsz=640,                # 输入尺寸和推理一致
    batch=1,                  # 单张图片只能设为1
    mosaic=0.0,               # 关闭Mosaic增强（仅1张图无法增强）
    workers=0,                # Windows下固定设0
    val=False,                # 关闭验证（仅1张图无验证意义）
    lr0=0.0001,               # 极低学习率，避免过拟合太快
    weight_decay=0.0001,      # 轻微正则化
    name='extreme_test_model' # 训练结果保存目录
)

print("\n🎉 极端训练完成！权重文件路径：runs/detect/extreme_test_model/weights/best.pt")