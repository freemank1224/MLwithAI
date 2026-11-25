# 图像识别课堂样本

每个 NPZ 文件都保留 28 × 28 的原始灰度像素（uint8）和原始标签。键名为 `x_train/y_train`、`x_valid/y_valid`、`x_test/y_test`。课堂按类别等量抽样，准确率反映这个均衡子集上的表现。

| 文件 | 类别 | 训练 | 验证 | 测试 | 来源与授权 |
| --- | ---: | ---: | ---: | ---: | --- |
| mnist.npz | 10 | 2,000 | 400 | 400 | MNIST © Yann LeCun、Corinna Cortes。子集沿用 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)。[Keras 数据说明](https://keras.io/api/datasets/mnist/) · [原始 NPZ](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) |
| sign_mnist.npz | 24 | 4,800 | 960 | 960 | tecperson 发布的 Sign Language MNIST，CC0。使用发布包中的 sign_mnist_train.csv 和 sign_mnist_test.csv。[原始数据与说明](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) |

两个数据集各自以 `np.random.default_rng(42)` 开始，按原始标签升序逐类抽样。每类先随机排列原训练文件的行号，再随机排列原测试文件的行号。取前者前 200 张训练、接下来 40 张验证，后者前 40 张测试。最后依次打乱训练、验证、测试子集各自的顺序并压缩保存。手势原始标签按 A=0、B=1 依次编码，跳过需要动作的 J=9 和 Z=25。笔记本在加载时转换为连续类别索引。
