import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
from model import CNN
from common import Adam
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问

# 1. 数据加载
def load_data(data_dir, img_size=(30, 30)):
    images, labels = [], []
    for label in range(43):
        class_dir = os.path.join(data_dir, "Train", str(label))
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)  # (C, H, W)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# 1. 加载数据
X, y = load_data("dataset")

# 2. 像素归一化到[0,1]范围
X = X / 255.0

# 3. 标签one-hot编码
y = to_categorical(y, num_classes=43)

# 4. 拆分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. 初始化模型和优化器
model = CNN(input_dim=(3, 30, 30), filter_num=30, filter_size=5,
            hidden_size=100, output_size=43)
optimizer = Adam(lr=0.001)


# 4. 训练循环
def train(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=32):
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        # 打乱数据
        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        epoch_loss, correct = 0, 0
        for i in tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # 前向传播
            y_pred = model.forward(X_batch)
            loss = model.last_layer.forward(y_pred, y_batch)
            epoch_loss += loss

            # 反向传播
            model.backward(1)
            optimizer.update(model.params, model.grads)

            # 计算准确率
            correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

        # 验证集评估
        val_pred = model.forward(X_val)
        val_loss = model.last_layer.forward(val_pred, y_val)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))

        # 记录结果
        history['loss'].append(epoch_loss / len(X_train))
        history['accuracy'].append(correct / len(X_train))
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        print(
            f"Epoch {epoch + 1} | Train Loss: {history['loss'][-1]:.4f} | Acc: {history['accuracy'][-1]:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return history

# 5. 执行训练
history = train(model, X_train, y_train, X_val, y_val)
import seaborn as sns
# 6. 测试集评估

def evaluate(model, test_dir):
    X_test, y_test = load_data(test_dir)
    X_test = X_test / 255.0
    y_pred = model.forward(X_test)
    # 将概率输出转换为类别预测
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_classes == y_test)
    test_cm = confusion_matrix(y_test, y_pred_classes)  # 使用类别预测
    sns.heatmap(test_cm, annot=True)
    print(f"Test Accuracy: {accuracy:.4f}")
evaluate(model, "dataset")
import matplotlib.pyplot as plt
import random
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)
def load_data_with_csv(data_dir, img_size=(30, 30)):
    """从CSV文件加载测试数据和标签"""
    import pandas as pd
    csv_path = os.path.join(data_dir, "Test.csv")#看这

    try:
        df = pd.read_csv(csv_path)
        print("CSV列名:", df.columns.tolist())  # 调试用，确认列名
    except Exception as e:
        print(f"错误：无法读取CSV文件 {csv_path}: {e}")
        return None, None, None

    images, labels, filenames = [], [], []

    for _, row in df.iterrows():
        img_relative_path = row['Path']
        label = row['ClassId']  # 类别ID

        # 构建完整图片路径
        img_path = os.path.join(data_dir, img_relative_path)

        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 - {img_path}")
            continue
        # 使用CSV中的ROI信息裁剪感兴趣区域（如果需要）
        x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        img = img[y1:y2 + 1, x1:x2 + 1]  # 裁剪ROI区域

        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        images.append(img)
        labels.append(label)
        filenames.append(os.path.basename(img_relative_path))  # 只保留文件名

    return np.array(images), np.array(labels), filenames


def visualize_random_prediction_with_label(model, test_dir):
    """随机测试图片预测可视化（带ROI处理和真实标签）"""
    X_test, y_test, filenames = load_data_with_csv(test_dir)
    if X_test is None or len(X_test) == 0:
        print("错误：没有加载到任何测试数据！")
        return

    X_test = X_test / 255.0  # 归一化

    # 随机选择一张图像
    idx = random.randint(0, len(X_test) - 1)
    test_image = X_test[idx]
    true_label = y_test[idx]
    filename = filenames[idx]

    # 模型预测
    y_pred = model.forward(test_image[np.newaxis, ...])
    pred_label = np.argmax(y_pred, axis=1)[0]
    pred_prob = np.max(y_pred, axis=1)[0]

    # 准备显示
    display_image = test_image.transpose(1, 2, 0)

    # 修正：将class_names定义为字典
    class_names = {
        0: "限速20", 1: "限速30", 2: "限速50", 3: "限速60",
        4: "限速70", 5: "限速80", 6: "解除限速80",
        7: "限速100", 8: "限速120", 9: "禁止超车",
        10: "卡车禁止超车", 11: "优先道路", 12: "让行",
        13: "停车", 14: "禁止通行", 15: "卡车禁止",
        16: "禁止进入", 17: "注意", 18: "危险弯道左",
        19: "危险弯道右", 20: "双弯道", 21: "颠簸路面",
        22: "湿滑路面", 23: "变窄", 24: "施工",
        25: "交通信号", 26: "行人", 27: "儿童",
        28: "自行车", 29: "冰雪", 30: "野生动物",
        31: "解除限制", 32: "右转", 33: "左转",
        34: "直行", 35: "直行或右转", 36: "直行或左转",
        37: "靠右", 38: "靠左", 39: "环岛",
        40: "解除超车限制", 41: "解除卡车超车限制", 42: "其他"
    }

    # 可视化
    plt.figure(figsize=(8, 8))
    plt.imshow(display_image)

    title = f"文件名: {filename}\n"
    title += f"真实类别: {class_names.get(int(true_label), str(true_label))} ({true_label})\n"  # 确保true_label是整数
    title += f"预测类别: {class_names.get(int(pred_label), str(pred_label))} ({pred_label})\n"


    plt.title(title, color='green' if true_label == pred_label else 'red')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return filename, true_label, pred_label

# 加载数据
X_test, y_test, filenames = load_data_with_csv("dataset")


filename, true_label, pred_label = visualize_random_prediction_with_label(model, "dataset")
print(f"文件名: {filename}")
print(f"真实类别ID: {true_label}, 预测类别ID: {pred_label}")
print("预测结果:", "✓ 正确" if true_label == pred_label else "✗ 错误")

