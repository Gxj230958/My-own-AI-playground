import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置环境变量以解决OpenMP警告
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置随机种子为680008
torch.manual_seed(680008)
np.random.seed(680008)

# 定义LeNet卷积神经网络模型
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一个卷积层，输入通道1，输出通道6，卷积核大小5x5
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # 第二个卷积层，输入通道6，输出通道16，卷积核大小5x5
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        # 第三个卷积层，输入通道16，输出通道120，卷积核大小5x5
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 120, kernel_size=5),
            torch.nn.ReLU(),
        )
        # 全连接层，将特征映射到输出类别
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(120, 84),
            torch.nn.Linear(84, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        # 第一层卷积+激活+池化
        x = self.conv1(x)
        # 第二层卷积+激活+池化
        x = self.conv2(x)
        # 第三层卷积+激活
        x = self.conv3(x)
        # 将特征图展平为一维向量
        x = x.view(batch_size, -1)  # (batch, 120,1,1) ==> (batch,120)
        # 全连接层处理
        x = self.fc(x)
        return x  # 输出10个类别的预测分数

# 定义简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 第一个全连接层，输入784（28x28像素），输出128
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        # 第二个全连接层，输入128，输出64
        self.fc2 = nn.Linear(128, 64)
        # 第三个全连接层，输入64，输出10（类别数）
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # 将输入展平为一维向量
        x = x.view(-1, 784)
        # 第一层全连接+激活
        x = self.fc1(x)
        x = self.relu(x)
        # 第二层全连接+激活
        x = self.fc2(x)
        x = self.relu(x)
        # 第三层全连接（输出层）
        x = self.fc3(x)
        return x

# 创建数据加载器
def create_data_loaders_lenet(batch_size=64):
    # 定义数据预处理和转换 - 适用于LeNet (32x32)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # LeNet的输入需要32x32的图像
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化处理
    ])

    # 加载训练数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 加载测试数据集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_data_loaders_simplenn(batch_size=64):
    # 定义数据预处理和转换 - 适用于SimpleNN (保持28x28)
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 标准化处理
    ])

    # 加载训练数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 加载测试数据集
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# 训练模型的函数
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    # 用于存储每个epoch的训练损失
    train_losses = []
    # 用于存储每个epoch的测试损失
    test_losses = []
    
    # 迭代训练指定的轮数
    for epoch in range(num_epochs):
        # 设置模型为训练模式
        model.train()
        running_loss = 0.0
        
        # 遍历训练数据集中的每个批次
        for images, labels in train_loader:
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累加损失
            running_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 计算测试集上的损失
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        # 计算平均测试损失
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
    
    return train_losses, test_losses

# 绘制损失曲线的函数
def plot_losses(train_losses, test_losses, model_name):
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Epochs-680008 ({model_name})')
    plt.legend()
    plt.savefig(f'{model_name}_train_loss_680008.png')
    print(f"Saved: {model_name}_train_loss_680008.png")
    plt.close()
    
    # 绘制测试损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Test Loss Over Epochs-680008 ({model_name})')
    plt.legend()
    plt.savefig(f'{model_name}_test_loss_680008.png')
    print(f"Saved: {model_name}_test_loss_680008.png")
    plt.close()

# 测试模型的函数
def evaluate_model(model, test_loader):
    # 设置模型为评估模式
    model.eval()
    correct = 0
    total = 0
    
    # 不计算梯度
    with torch.no_grad():
        for images, labels in test_loader:
            # 前向传播
            outputs = model(images)
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 累加样本总数
            total += labels.size(0)
            # 累加正确预测的样本数
            correct += (predicted == labels).sum().item()
    
    # 计算准确率
    accuracy = 100 * correct / total
    print(f'Accuracy of the {model.__class__.__name__} on the test images: {accuracy:.2f}%')
    return accuracy

# 推理模型的函数，并输出测试图片
def infer_model_and_display_image(model, test_loader):
    # 设置模型为评估模式
    model.eval()
    # 获取一个样本
    sample_input, true_label = next(iter(test_loader))
    sample_input = sample_input[:1]  # 取第一个样本
    true_label = true_label[:1]

    # 不计算梯度
    with torch.no_grad():
        # 前向传播
        output = model(sample_input)
        # 获取预测结果
        _, predicted = torch.max(output.data, 1)

    # 显示图片和预测结果
    plt.figure(figsize=(5, 2))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_input.squeeze().numpy(), cmap='gray')
    plt.title(f'True Label: {true_label.item()}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f'Predicted Label: {predicted.item()}', fontsize=12, ha='center', va='center')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{model.__class__.__name__}_prediction_680008.png')
    print(f"Saved: {model.__class__.__name__}_prediction_680008.png")
    plt.close()

    return predicted.item()

# 主函数，整合训练、测试和推理
def main():
    # 设置超参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # 创建数据加载器
    train_loader_lenet, test_loader_lenet = create_data_loaders_lenet(batch_size=batch_size)
    train_loader_simplenn, test_loader_simplenn = create_data_loaders_simplenn(batch_size=batch_size)

    # 训练和测试LeNet模型
    print("Training LeNet model...")
    lenet_model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet_model.parameters(), lr=learning_rate)
    lenet_train_losses, lenet_test_losses = train_model(lenet_model, train_loader_lenet, test_loader_lenet, criterion, optimizer, num_epochs=num_epochs)
    
    # 绘制LeNet的损失曲线
    plot_losses(lenet_train_losses, lenet_test_losses, "LeNet")
    
    # 测试LeNet模型
    lenet_accuracy = evaluate_model(lenet_model, test_loader_lenet)
    
    # 推理LeNet模型并显示图片
    lenet_prediction = infer_model_and_display_image(lenet_model, test_loader_lenet)

    # 训练和测试SimpleNN模型
    print("\nTraining SimpleNN model...")
    simplenn_model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(simplenn_model.parameters(), lr=learning_rate)
    simplenn_train_losses, simplenn_test_losses = train_model(simplenn_model, train_loader_simplenn, test_loader_simplenn, criterion, optimizer, num_epochs=num_epochs)
    
    # 绘制SimpleNN的损失曲线
    plot_losses(simplenn_train_losses, simplenn_test_losses, "SimpleNN")
    
    # 测试SimpleNN模型
    simplenn_accuracy = evaluate_model(simplenn_model, test_loader_simplenn)
    
    # 推理SimpleNN模型并显示图片
    simplenn_prediction = infer_model_and_display_image(simplenn_model, test_loader_simplenn)
    
    # 比较两个模型的性能
    print("\nModel Comparison:")
    print(f"LeNet Accuracy: {lenet_accuracy:.2f}%")
    print(f"SimpleNN Accuracy: {simplenn_accuracy:.2f}%")

if __name__ == "__main__":
    # 打印当前工作目录，帮助找到保存的图片
    print("Current working directory:", os.getcwd())
    main() 