from ultralytics import YOLO
import cv2  # 需安装：pip install opencv-python
import numpy as np

# 加载训练好的模型
model = YOLO('runs/detect/my_custom_model/weights/best.pt')

# 对单张图片推理
results = model('1.png')  # 你的测试图片
result = results[0]  # 取出列表中的结果对象

# 方案1：生成带检测框的图片并保存（推荐，无弹窗问题）
annotated_img = result.plot()  # 生成标注后的图片（numpy数组）
cv2.imwrite('annotated_result.png', annotated_img)  # 保存到本地
print("✅ 标注后的图片已保存为 annotated_result.png，可直接打开查看")

# 方案2（可选）：弹出窗口显示图片（需确保有图形界面）
# cv2.imshow('Detection Result', annotated_img)
# cv2.waitKey(0)  # 按任意键关闭窗口
# cv2.destroyAllWindows()

# 查看检测详情（帮助排查无检测结果的问题）
print(f"\n📊 检测详情：")
print(f"图片原始尺寸：{result.orig_shape}")
print(f"检测到的目标数量：{len(result.boxes)}")
if len(result.boxes) == 0:
    print("⚠️  未检测到任何目标，可能原因：")
    print("  1. 训练数据过少（仅1张），模型未学到有效特征")
    print("  2. 测试图片1.png中无训练标注的类别目标")
    print("  3. 模型训练轮数不足，或类别ID/标注格式错误")