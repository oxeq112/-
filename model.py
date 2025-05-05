from common import SoftmaxWithLoss, Relu, Adam, Affine, Convolution, Pooling, softmax
import numpy as np


class CNN:
    def __init__(
        self,
        input_dim=(3, 30, 30),
        filter_num=30,
        filter_size=5,
        hidden_size=100,
        output_size=43,
        weight_init_std=0.01,
    ):
        """
        初始化CNN网络参数和层
        参数:
            input_dim: 输入数据的维度 (通道, 高, 宽)
            filter_num: 卷积层的滤波器数量
            filter_size: 滤波器大小
            hidden_size: 全连接层隐藏单元数
            output_size: 输出层大小（类别数）
            weight_init_std: 权重初始化标准差
        """
        # 网络参数初始化
        self.params = {}
        # 卷积层1参数
        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params["b1"] = np.zeros(filter_num)
        # 卷积层2参数
        self.params["W2"] = weight_init_std * np.random.randn(
            filter_num, filter_num, filter_size, filter_size
        )
        self.params["b2"] = np.zeros(filter_num)

        # 计算卷积和池化后的输出尺寸
        conv1_output_size = (
            input_dim[1] - filter_size + 2 * 0
        ) // 1 + 1  # 卷积层1输出尺寸
        pool1_output_size = conv1_output_size // 2  # 池化层1输出尺寸
        conv2_output_size = (
            pool1_output_size - filter_size + 2 * 0
        ) // 1 + 1  # 卷积层2输出尺寸
        pool2_output_size = conv2_output_size // 2  # 池化层2输出尺寸

        # 全连接层输入维度
        fc_input_size = filter_num * pool2_output_size * pool2_output_size

        # 全连接层1参数
        self.params["W3"] = weight_init_std * np.random.randn(
            fc_input_size, hidden_size
        )
        self.params["b3"] = np.zeros(hidden_size)
        # 输出层参数
        self.params["W4"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b4"] = np.zeros(output_size)

        # 网络层定义
        self.layers = {}
        self.layers["Conv1"] = Convolution(
            self.params["W1"], self.params["b1"], stride=1, pad=0
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Conv2"] = Convolution(
            self.params["W2"], self.params["b2"], stride=1, pad=0
        )
        self.layers["Relu2"] = Relu()
        self.layers["Pool2"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W3"], self.params["b3"])
        self.layers["Relu3"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W4"], self.params["b4"])
        self.flatten_size = None  # 用于记录展平后的尺寸

        # 最后一层（SoftmaxWithLoss在训练时单独添加）
        self.last_layer = SoftmaxWithLoss()

    def forward(self, x):
        # 确保输入始终是4D
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)  # 单样本添加batch维度

        # 卷积层部分
        for layer in ["Conv1", "Relu1", "Pool1", "Conv2", "Relu2", "Pool2"]:
            x = self.layers[layer].forward(x)

        # 展平操作
        batch_size = x.shape[0]
        # 保存展平前的形状
        self.flatten_shape = x.shape
        # 执行展平
        x = x.reshape(batch_size, -1)  # 保持batch维度

        # 全连接部分
        for layer in ["Affine1", "Relu3", "Affine2"]:
            x = self.layers[layer].forward(x)

        return x

    def backward(self, dout):
        # 调用 SoftmaxWithLoss 的反向传播，获取正确的 dout 形状
        dout = self.last_layer.backward(dout)
        # 按反向顺序传播
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            # 如果 dout 是二维向量且元素数量与展平前一致，就转换回四维张量
            if dout.ndim == 2 and dout.size == np.prod(self.flatten_shape):
                dout = dout.reshape(self.flatten_shape)
            dout = layer.backward(dout)
        return dout

    def predict(self, x):
        """
        预测（前向传播 + Softmax）
        """
        x = self.forward(x)
        return softmax(x)

    def loss(self, x, t):
        """
        计算损失（前向传播 + SoftmaxWithLoss）
        参数:
            x: 输入数据
            t: 真实标签
        """
        y = self.forward(x)
        return self.last_layer.forward(y, t)

    @property
    def grads(self):
        """收集网络中所有层参数的梯度"""
        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Conv2"].dW
        grads["b2"] = self.layers["Conv2"].db
        grads["W3"] = self.layers["Affine1"].dW
        grads["b3"] = self.layers["Affine1"].db
        grads["W4"] = self.layers["Affine2"].dW
        grads["b4"] = self.layers["Affine2"].db
        return grads
