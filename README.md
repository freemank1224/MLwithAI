# MLwithAI：在实验中理解机器学习

这是一套面向机器学习与编程初学者的 Jupyter Notebook 教学材料。每一课先用情景图或算法示意图提出一个具体问题，再做预测、观察交互图、编写代码并验证结果。课尾的「NumPy 手写实现」是选学内容，逐步拆解算法的关键计算；课程中的现成库主要用于对照。基础课多使用教学模拟数据，深度学习专题使用注明来源的真实图像子集。

## 安装与启动

三节深度学习课已在 Python 3.10.15、PyTorch 2.5.1、Apple 芯片 Mac 的 CPU 上完整运行。项目依赖锁定为 `torch==2.5.1`。建议机房使用 64 位 Python 3.10，在项目根目录打开终端。

Win11 有 NVIDIA 显卡且首次配置课程环境的同学，可以按[全新 CUDA 环境](#win11-全新-cuda-环境)安装。已有环境的同学可以直接用对应内核运行 CNN，遇到内核退出时优先[在原环境排查和修复](#win11-已有环境的-mklopenmp-冲突排查)。

### Win11 全新 CUDA 环境

以下命令适用于 Win11 x64 和 NVIDIA 显卡。在刚克隆的项目根目录打开普通 PowerShell。如果提示符有 `(base)` 或其他 Conda 环境名，先运行 `conda deactivate`，直到退出 Conda 环境。需要独立安装的 x64 CPython 3.10 和可运行的 NVIDIA 驱动。如果缺少 Python，可安装官方页面的 **Windows installer (64-bit)** 并勾选 Python Launcher，安装后重新打开 PowerShell。可用 `py -0p` 查看解释器路径。[Python 3.10 Windows 安装包](https://www.python.org/downloads/release/python-31011/)

在同一个 PowerShell 窗口依次运行下面两段。第一段按驱动公布的 CUDA 版本选择 PyTorch 2.5.1 的官方构建，在独立 `.venv` 中安装依赖并注册专用内核。最终能否用该构建在显卡上训练，以第二段的实跑结果为准。[PyTorch 2.5.1 官方安装表](https://pytorch.org/get-started/previous-versions/#v251)

```powershell
$ErrorActionPreference = 'Stop'
$venvPath = '.venv'
$kernelName = 'mlwithai-win11-cuda'
$kernelDisplayName = 'Python (MLwithAI Win11 CUDA)'

if ($env:CONDA_PREFIX) { throw 'Exit the Conda environment before continuing.' }
if (!(Test-Path 'requirements.txt') -or !(Test-Path 'MachineLearning\CNNImageRecognition.ipynb')) {
    throw 'Run these commands from the MLwithAI project root.'
}
if (Test-Path $venvPath) { throw "Choose a new environment path: $venvPath already exists." }

$basePython = py -3.10 -I -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) { throw 'Install standalone x64 CPython 3.10 first.' }
& $basePython -I -c "import platform, sys; assert sys.version_info[:2] == (3, 10) and platform.machine().upper() == 'AMD64' and sys.maxsize > 2**32 and sys.prefix == sys.base_prefix; print(sys.executable, sys.version)"
if ($LASTEXITCODE -ne 0 -or (Test-Path (Join-Path (Split-Path -Parent $basePython) 'conda-meta'))) {
    throw 'Use a standalone x64 CPython 3.10 installation.'
}

$gpuInfo = nvidia-smi
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed. Check the NVIDIA driver.' }
$gpuInfo
$cudaMatch = [regex]::Match(($gpuInfo -join "`n"), 'CUDA Version:\s*(\d+\.\d+)')
if (!$cudaMatch.Success) { throw 'Could not read CUDA Version from nvidia-smi.' }
$driverCuda = [version]$cudaMatch.Groups[1].Value
if ($driverCuda -ge [version]'12.4') {
    $cudaBuild = 'cu124'
} elseif ($driverCuda -ge [version]'12.1') {
    $cudaBuild = 'cu121'
} elseif ($driverCuda -ge [version]'11.8') {
    $cudaBuild = 'cu118'
} else {
    throw 'Check driver compatibility with the CUDA builds for PyTorch 2.5.1.'
}

# 清除当前 PowerShell 进程继承的 Python 路径和 OpenMP 绕过设置。
Remove-Item Env:PYTHONHOME, Env:PYTHONPATH, Env:KMP_DUPLICATE_LIB_OK -ErrorAction SilentlyContinue
& $basePython -I -m venv $venvPath
if ($LASTEXITCODE -ne 0) { throw 'Could not create the course environment.' }
$coursePython = (Resolve-Path (Join-Path $venvPath 'Scripts\python.exe')).Path

function Invoke-CoursePython {
    & $coursePython -I @args
    if ($LASTEXITCODE -ne 0) { throw "Course Python exited with code $LASTEXITCODE." }
}

Invoke-CoursePython -m pip --isolated install --index-url "https://download.pytorch.org/whl/$cudaBuild" "torch==2.5.1+${cudaBuild}"
Invoke-CoursePython -m pip --isolated install --index-url https://pypi.org/simple '--only-binary=numpy,scipy,matplotlib' -r requirements.txt
Invoke-CoursePython -m pip check
Invoke-CoursePython -c "import sys; from jupyter_client.kernelspec import KernelSpecManager; assert sys.argv[1].lower() not in KernelSpecManager().find_kernel_specs(), 'Kernel name already exists. Choose a new kernel name.'" $kernelName
Invoke-CoursePython -m ipykernel install --user --name $kernelName --display-name $kernelDisplayName
```

第二段通过刚注册的内核执行整本 CNN notebook，包含第一次样本绘图和 GPU 训练。终端需要输出课程解释器路径、CUDA 版 PyTorch、显卡名称、`cuda:0` 和样本图检查结果。AI Agent 可直接在终端验收，执行结果保存在内存中。

```powershell
@'
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

kernel_name, cuda_build = sys.argv[1:]
expected_torch = "2.5.1+" + cuda_build
path = Path("MachineLearning/CNNImageRecognition.ipynb").resolve()
notebook = nbformat.read(path, as_version=4)
notebook.cells.append(nbformat.v4.new_code_cell(f"""
from pathlib import Path
import sys

assert Path(sys.executable).resolve() == Path({sys.executable!r}).resolve(), "Wrong Jupyter kernel"
assert torch.__version__ == {expected_torch!r}, torch.__version__
assert torch.cuda.is_available() and device.type == "cuda", "CUDA training is unavailable"
model, _ = runs["MNIST", "CNN"]
actual_device = next(model.parameters()).device
assert str(actual_device) == "cuda:0", actual_device
print("Kernel:", sys.executable)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(actual_device), actual_device)
"""))
NotebookClient(
    notebook, kernel_name=kernel_name, timeout=300,
    resources={"metadata": {"path": str(path.parent)}},
).execute()
first_image = next(cell for cell in notebook.cells if "ax.imshow(img," in cell.source)
assert any("image/png" in item.get("data", {}) for item in first_image.outputs), "First plot is missing"
for item in notebook.cells[-1].outputs:
    if item.output_type == "stream":
        print(item.text, end="")
print("CNN notebook completed. First sample plot rendered.")
'@ | & $coursePython -I - $kernelName $cudaBuild
if ($LASTEXITCODE -ne 0) { throw 'CNN notebook validation failed. Check the error above.' }
```

开始上课时，在同一窗口运行 `& $coursePython -I -m jupyterlab`，打开 notebook 后选择 `$kernelDisplayName` 指定的内核，默认是 **Python (MLwithAI Win11 CUDA)**。以后可在项目根目录运行 `& .\.venv\Scripts\python.exe -I -m jupyterlab`。如使用其他虚拟环境路径，请用那个环境的解释器启动。

如果验收时报 `CUDA training is unavailable` 或 `no kernel image is available`，保留显卡型号、驱动版本和完整错误，检查这块显卡是否受所用构建支持。需要先上课时，可用下文的 CPU 路线另建环境。

### Win11 已有环境的 MKL/OpenMP 冲突排查

#### 确认出错的内核和库

在出错的 notebook 中重启内核，先单独运行 `import sys; print(sys.executable)`。在 PowerShell 激活该解释器对应的 Conda 环境，再运行以下命令。两处 Python 路径应一致。先清除当前终端可能继承的 OpenMP 绕过设置，让原始错误可以正常出现。

```powershell
Remove-Item Env:KMP_DUPLICATE_LIB_OK -ErrorAction SilentlyContinue
python -c "import sys; print(sys.executable)"
conda list --show-channel-urls '^(numpy(-base)?|scipy|blas|libblas|mkl([_-].*)?|intel-openmp|llvm-openmp|pytorch(-cuda)?|torch|matplotlib(-base)?)$'
python -m jupyterlab --debug
```

重跑到出错单元。若日志显示 `OMP: Error #15`，并列出 `libiomp5md.dll` 或 `libomp.dll`，可确认是同一进程中的 OpenMP 运行库冲突。Conda 和 pip 可以在一个环境里安装不同的包。一个常见冲突组合是启用 MKL 的 NumPy 与自带 OpenMP 的 PyTorch，实际来源要以包清单和库加载情况为准。如果日志只有 `Kernel died`，保留出错单元和退出前的日志，先继续定位。CNN 首张样本图使用 CPU 数据。[PyTorch 的 Windows 同类案例](https://github.com/pytorch/pytorch/issues/167979)

可以用下面两条命令在**独立进程**中分别检查 NumPy 和 PyTorch。若输出包含 `filepath`，记录 BLAS/OpenMP 库的路径和所用后端。它们显示各自加载的运行库，有助于区分 Conda 的 `Library\bin`、PyTorch 的 `torch\lib` 以及环境外的路径。

```powershell
python -X faulthandler -u -c "import numpy as n; a=n.ones((128,128)); a@a; from threadpoolctl import threadpool_info; print(threadpool_info())"
python -X faulthandler -u -c "import torch; a=torch.ones((128,128)); a@a; from threadpoolctl import threadpool_info; print(threadpool_info())"
```

#### 保留现有 Conda 环境原地修复

如果已确认是这种冲突，而且 NumPy/SciPy 一侧使用 MKL，可以先关闭 JupyterLab，在**同一环境**中做一次可逆测试。oneMKL 如果使用可切换的动态接口，`SEQUENTIAL` 会让它走单线程计算；其他加载 OpenMP 的组件仍需单独检查。此设置仅在本次 JupyterLab 启动期间生效，相关 MKL CPU 计算可能变慢。[Intel oneMKL 线程层说明](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2025-1/call-onemkl-functions-from-multi-threaded-apps.html)

```powershell
$previousMklLayer = $env:MKL_THREADING_LAYER
try {
    $env:MKL_THREADING_LAYER = 'SEQUENTIAL'
    python -m jupyterlab --debug
} finally {
    $env:MKL_THREADING_LAYER = $previousMklLayer
}
```

需要在正常启动方式下长期使用时，可在现有环境里让 Conda 把 NumPy/SciPy 的 BLAS 后端切到 OpenBLAS，保留已有的 CUDA 版 PyTorch。先关闭 JupyterLab 和正在使用该环境的 Python 进程。按包清单里 **NumPy/SciPy 所用的渠道**选下面一条预演，并记下当前 Conda 修订号。

```powershell
conda list --revisions

# NumPy/SciPy 使用 conda-forge，且环境已有 libblas。
conda install --dry-run "libblas=*=*_openblas"

# NumPy/SciPy 使用 defaults，且环境用 blas 元包选择后端。
conda install --dry-run "blas=*=openblas"
```

确认安装计划把相关数值库切到 OpenBLAS，并保留 Python 版本、CUDA PyTorch 和其他项目需要的包，再对选中的命令去掉 `--dry-run` 执行。若求解失败或要大范围更换软件包和渠道，保留原环境和预演输出，继续核对是哪个包加载了第二套 OpenMP。此处的回滚适用于 Conda 的包变更，可用 `conda install --revision N` 恢复到记录的修订号。[conda-forge 的 BLAS 切换说明](https://conda-forge.org/docs/maintainer/knowledge_base/#switching-blas-implementation) · [Anaconda 的 Windows OpenBLAS 包](https://anaconda.org/anaconda/blas/files) · [Conda 修订与回滚](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#restoring-an-environment)

修复后从同一环境重新启动 JupyterLab，选原内核。先确认首张样本图正常显示，再跑完整本 CNN。需要 GPU 的机器同时按下文[启动并验证](#启动并验证)检查 CUDA 卷积与反向传播。保留原有 DLL，由包管理器维护；`KMP_DUPLICATE_LIB_OK=TRUE` 会绕过重复初始化检查，可能导致崩溃或计算错误。[Intel OpenMP 重复加载说明](https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/Windows-libiomp5md-reports-multiple-initialisation/m-p/1361874)

### 创建课程环境

macOS 或 Linux：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Windows CPU 环境（PowerShell）：

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

### NVIDIA 显卡安装（Linux）

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
