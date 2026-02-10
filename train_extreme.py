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
# results = model.train(
#     data='./project/data.yaml',       # 你的配置文件路径
#     epochs=500,               # 大幅增加训练轮数，强制模型记住单张图
#     imgsz=640,                # 输入尺寸和推理一致
#     batch=1,                  # 单张图片只能设为1
#     mosaic=1.0,               # 关闭Mosaic增强（仅1张图无法增强）
#     workers=0,                # Windows下固定设0
#     val=False,                # 关闭验证（仅1张图无验证意义）
#     lr0=0.0001,               # 极低学习率，避免过拟合太快
#     weight_decay=0.0001,      # 轻微正则化
#     name='extreme_test_model' # 训练结果保存目录
# )

results = model.train(
    data='./project6/data.yaml',   # 配置文件路径
    epochs=150,                   # 4张图训100轮足够，500轮会严重过拟合
    imgsz=960,                    # 输入尺寸和推理一致
    batch=1,                      # 4张图仍建议batch=1（稳定），也可试batch=2
    mosaic=0.0,                   # 4张图刚好满足Mosaic增强的最低要求（开启！）
    workers=0,                    # Windows下固定设0
    val=False,                    # 4张图暂不验证（后续≥8张再开）
    lr0=0.0001,                   # 学习率比单张图稍高，适配多图
    weight_decay=0.0001,          # 适度正则化，防止过拟合
    name='new_model6'            # 训练结果保存目录
)

print("\n🎉 极端训练完成！权重文件路径：runs/detect/extreme_test_model/weights/best.pt")