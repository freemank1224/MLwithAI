# MLwithAI：在实验中理解机器学习

这是一套面向机器学习与编程初学者的 Jupyter Notebook 教学材料。每一课先用情景图或算法示意图提出一个具体问题，再做预测、观察交互图、编写代码并验证结果。课尾的「NumPy 手写实现」是选学内容，逐步拆解算法的关键计算；课程中的现成库主要用于对照。基础课多使用教学模拟数据，深度学习专题使用注明来源的真实图像子集。

## 安装与启动

三节深度学习课已在 Python 3.10.15、PyTorch 2.5.1、Apple 芯片 Mac 的 CPU 上完整运行。项目依赖锁定为 `torch==2.5.1`。建议机房使用 64 位 Python 3.10，在项目根目录打开终端。

### 创建课程环境

macOS 或 Linux：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

已经创建过环境的同学只需激活它。根据要使用的设备，从下面两种安装方式中选一种。

### CPU 安装

Apple 芯片 Mac 可直接运行 `python -m pip install -r requirements.txt`。Windows 或 Linux 的 x86-64 机器可先安装官方 CPU 构建，再补齐课程依赖：

```bash
python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### NVIDIA 显卡安装（Windows 或 Linux）

在终端运行 `nvidia-smi`，确认系统能列出显卡和驱动版本。根据显卡型号及驱动支持情况选择 PyTorch 2.5.1 的 CUDA 构建。下面以 **CUDA 12.4** 为例；选用 CUDA 12.1 时，将命令中的两处 `cu124` 都换成 `cu121`。选用 CUDA 11.8 时都换成 `cu118`。包名中的构建标记也能让 pip 区分现有的同版本 CPU 安装。[PyTorch 2.5.1 官方安装表](https://pytorch.org/get-started/previous-versions/#v251)

```bash
python -m pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

### 启动并验证

两种环境使用相同的启动命令。首次安装后注册内核，之后上课只需激活环境并启动 JupyterLab。

```bash
python -m ipykernel install --user --name mlwithai --display-name "Python (MLwithAI)"
python -m jupyterlab
```

打开课程 notebook，选择 **Python (MLwithAI)** 内核。下格在所选设备执行一次卷积和反向传播。**CPU 安装保留 `"cpu"`，NVIDIA 安装改为 `"cuda"`**，这样能明确检查本次安装目标。

```python
import sys
print("当前内核", sys.executable)

import torch
target = "cpu"
print("PyTorch", torch.__version__, "CUDA 构建", torch.version.cuda)
assert torch.__version__.split("+")[0] == "2.5.1", "请检查课程环境中的 PyTorch 版本"
if target == "cuda":
    assert torch.cuda.is_available(), "请检查当前内核、CUDA 构建和 NVIDIA 驱动"
    print("显卡", torch.cuda.get_device_name(0))

layer = torch.nn.Conv2d(1, 8, 3).to(target)
output = layer(torch.ones(1, 1, 28, 28, device=target))
output.mean().backward()
assert layer.weight.grad.isfinite().all().item()
print("卷积与反向传播通过", output.device, tuple(output.shape))
```

CPU 验证应显示 `cpu (1, 8, 26, 26)`，NVIDIA 验证应显示显卡名称和 `cuda:0 (1, 8, 26, 26)`。课程准备单元会自动选择可用的 CUDA 设备，否则使用 CPU。如果遇到 `No module named 'torch'`，核对第一行的内核路径，切回课程内核；若课程环境确实缺包，按上面选定的路线安装一次，重启内核即可。

本课程不依赖联网服务或 API 密钥。原项目可选的 Jupyter AI、XGBoost、LightGBM 和 pandas 不属于运行这些课程的基础依赖；有需要时可在同一环境另行安装，例如 `python -m pip install 'jupyter-ai[all]' xgboost lightgbm pandas`。

三课训练时每轮刷新训练集和验证集的交叉熵（loss）曲线。训练损失按该轮实际送入模型的样本求平均。课程最后用各轮权重回算原始测试集的损失，将训练和测试 loss 分图展示，共用纵轴刻度。模型选择依据验证集，测试曲线用于最终复盘。

## 推荐学习顺序

| 顺序 | 主题 | 要解决的问题 |
| --- | --- | --- |
| 0 | [Python 基础练习](MachineLearning/Execises/BasicTest.ipynb) | 用变量、列表和函数处理一小组数据，为后续代码练习做准备 |
| 1 | [线性回归](MachineLearning/LinearReg.ipynb) | 怎样用一条直线预测连续数值？梯度下降如何改进这条线？ |
| 2 | [逻辑回归](MachineLearning/LogisticReg.ipynb) | 怎样估计二分类概率？调整决策阈值会改变什么？ |
| 3 | [决策树、随机森林与梯度提升树](MachineLearning/DecisionTrees.ipynb) | 从挑西瓜开始，理解树如何划分数据，以及随机森林、梯度提升树与 XGBoost 的联系和区别。 |
| 4 | [K-means 聚类](MachineLearning/Kmeans.ipynb) | 没有标签时，如何反复分配样本并更新中心？ |
| 5 | [层次聚类](MachineLearning/CascadedClustering.ipynb) | 如何从单个样本开始逐步合并，并用树状图选择分组？ |
| 6 | [主成分分析 PCA](MachineLearning/PCA.ipynb) | 怎样投影到较少维度，同时尽量保留数据的变化？ |
| 7 | [Q-learning](RLsimple.ipynb) | 智能体如何通过奖励和探索在网格中学会行动？ |

### 深度学习系列专题

| 任务 | 实验与交付 |
| --- | --- |
| 01 · [构建简单 CNN 进行图像识别](MachineLearning/CNNImageRecognition.ipynb) | 建议先学逻辑回归。从像素和滑窗乘加出发，分别在 MNIST 手写数字和 Sign Language MNIST 静态手势字母上训练 FCN 与 CNN。用四组测试成绩、参数量和具体错例解释自己的模型选择。 |
| 02 · [图像增强：从一张照片生成训练视图](MachineLearning/ImageAugmentation.ipynb) | 调整旋转、平移和亮度，检查手势标签是否仍成立。在相同预算下训练三组 CNN，用原图与固定扰动的成绩为一个拍摄场景选择配方。 |
| 03 · [迁移学习：从手写数字到静态手势](MachineLearning/TransferLearning.ipynb) | 先用本地 MNIST 预训练，再给手势任务换头。在每类 20 张训练图的条件下比较从头训练、冻结特征和微调，用学习曲线、权重变化和测试准确率解释选择。 |

每个笔记本都可以单独学习；线性回归和基础练习适合先学。课中请从上到下逐格运行，遇到问题先作预测、操作演示，再回到代码改变参数或补充实现，并运行紧接着的检查。标为「演示」或「准备」的代码直接运行即可，无需理解或修改 Widget；标为「练习」的代码才是需要关注的算法实现。检查通过只说明满足所给例子，还要观察图形并思考不同输入下的行为。

部分演示控件的实现位于 `MachineLearning/teaching_widgets.py`，情景插图位于 `assets/`，都是课程配套文件。请保留整个项目目录结构，不要只复制单个笔记本。修改练习代码后，按笔记本中的提示重新运行相应的训练、演示和检查单元。

决策树课程在关键阶段安排了互动自测：运行题目单元，选择答案并提交后查看解释；可以修改答案再次提交。自测不影响后续代码。

课程还提供三个可独立重置的图形互动实验：亲手选择问题建立决策树、抽样组合小森林、逐轮修正甜度预测。图形会随分裂、抽样和预测实时更新。它们使用教学模拟数据，运行对应单元后即可操作；需要保留 `MachineLearning/watermelon_tree_lab.py` 和 `MachineLearning/watermelon_ensemble_labs.py`。

## 使用交互图

使用工具栏的运行按钮或 `Shift + Enter` 执行单元格。图上的滑块、播放按钮或其他控件可用于逐步查看计算；先运行其上方的数据和函数单元格，再运行控件单元格。修改了前面的代码后，需要重新运行受影响的后续单元格。若状态混乱，选择 **Kernel → Restart Kernel and Run All Cells**，然后重新操作控件。自动播放有助于看到变化，暂停后逐步查看更适合比较每次更新。

如果控件显示为纯文本或无法响应：

1. 确认是在 JupyterLab 中打开 notebook，而不是仅在 GitHub 或静态预览中查看；静态预览不能执行 Python 或响应控件。
2. 确认当前 notebook 右上角的内核是安装依赖的 Python 环境。在 notebook 中运行 `import sys; print(sys.executable)`，再在终端用该解释器执行 `python -m pip install -r requirements.txt`（将 `python` 替换为显示的解释器路径）。
3. 重启内核并刷新浏览器页面，然后从头顺序运行。若仍看不到控件，在该环境检查 `python -m pip show ipywidgets jupyterlab ipykernel`；必要时重新安装依赖并重新启动 JupyterLab。
4. 图形中的中文若显示为方框，请在系统安装中文字体并重启内核；具体字体取决于操作系统。无中文字体不影响算法数值。

`MachineLearning/Manuscript.md` 是旧课程草稿，包含尚未完成的章节和未经校正的示例；请以实际 notebook 的内容为准。
