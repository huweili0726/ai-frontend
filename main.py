from ultralytics import YOLO
import os

# 加载预训练模型
# model = YOLO('yolov8n.pt')  # 也可以用 yolov8s.pt / yolov8m.pt 等
# 检查本地是否已有权重文件
weight_path = "yolov8n.pt"
if os.path.exists(weight_path):
    print(f"✅ 本地已存在 {weight_path}，直接加载")
else:
    print(f"📥 本地无 {weight_path}，开始下载")

# 加载模型（此时只会在文件不存在时下载）
model = YOLO(weight_path)

# 开始训练
# results = model.train(
#     data='./project/data.yaml',  # 配置文件路径
#     epochs=100,          # 训练轮数
#     imgsz=640,           # 输入图片尺寸
#     batch=16,            # 批次大小
#     name='my_custom_model'  # 训练结果保存的文件夹名
# )


# 开始训练（修改关键参数）
results = model.train(
    data='./project/data.yaml',  # 配置文件路径
    epochs=100,          # 训练轮数
    imgsz=640,           # 输入图片尺寸
    batch=1,             # 批次大小改为1（匹配仅1张图片的数据集）【当图片数≥16 张时，可将 batch=1 改回 batch=16】
    name='my_custom_model',
    mosaic=0.0,          # 关闭Mosaic数据增强（核心修复项）【当图片数≥4 张时，可将 mosaic=0.0 改回默认的 mosaic=1.0】
    workers=0,           # 关闭多线程加载（Windows下少量数据易出问题）
    val=False            # 若只有1张图，暂时关闭验证（避免验证集也报同样错）
)