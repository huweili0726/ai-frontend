from ultralytics import YOLO
import cv2
import numpy as np

# 加载极端训练后的模型
model = YOLO('runs/detect/new_model6/weights/best.pt')

# 用同一张图1.png推理
results = model('1.png')
result = results[0]

# 过滤置信度>=0.85的检测框
mask = result.boxes.conf >= 0.85
result.boxes = result.boxes[mask]

# 生成并保存标注图片
annotated_img = result.plot()
cv2.imwrite('extreme_test_result6.png', annotated_img)

# 打印详细检测结果
print("📊 极端测试检测结果：")
print(f"图片尺寸：{result.orig_shape}")
print(f"检测到的目标数量：{len(result.boxes)}")

if len(result.boxes) > 0:
    # 打印检测到的目标信息（类别、置信度、坐标）
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()  # 像素坐标：x1,y1,x2,y2
        if confidence >= 0.85:
            print(f"✅ 检测到：{cls_name}（置信度：{confidence:.2f}），坐标：{xyxy}")
    print("\n🎉 测试成功！模型已识别到目标，说明标注/配置无问题！")
else:
    print("\n❌ 仍未检测到目标，大概率是标注/配置错误：")
    print("  1. 检查1.txt标注文件的class_id是否和data.yaml的nc匹配；")
    print("  2. 检查标注文件的坐标是否为0-1的归一化值；")
    print("  3. 检查1.png和1.txt是否在正确的images/labels目录下；")
    print("  4. 检查data.yaml的path路径是否正确。")