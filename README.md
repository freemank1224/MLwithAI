# MLwithAI：在实验中理解机器学习

这是一套面向机器学习与编程初学者的 Jupyter Notebook 教学材料。每一课先用三格情景图提出一个具体问题，再做预测、观察交互图、编写代码并验证结果。课尾的「NumPy 手写实现」是选学内容，逐步拆解算法的关键计算；课程中的现成库主要用于对照。图中的数据是教学用模拟数据，结论不应直接用于真实决策。

## 安装与启动

需要 Python 3 和可以安装 Python 软件包的环境。在项目根目录打开终端，建议建立独立环境：

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS / Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name mlwithai --display-name "Python (MLwithAI)"
python -m jupyterlab
```

浏览器打开 JupyterLab 后，从左侧文件浏览器进入相应的 `.ipynb`，选择 **Python (MLwithAI)** 内核。下一次打开项目时，先激活同一个虚拟环境，再运行 `python -m jupyterlab`。安装只需一次；若更新了依赖，请重新运行安装命令。

本课程不依赖联网服务或 API 密钥。原项目可选的 Jupyter AI、XGBoost、LightGBM 和 pandas 不属于运行这些课程的基础依赖；有需要时可在同一环境另行安装，例如 `python -m pip install 'jupyter-ai[all]' xgboost lightgbm pandas`。

## 推荐学习顺序

| 顺序 | 主题 | 要解决的问题 |
| --- | --- | --- |
| 0 | [Python 基础练习](MachineLearning/Execises/BasicTest.ipynb) | 用变量、列表和函数处理一小组数据，为后续代码练习做准备 |
| 1 | [线性回归](MachineLearning/LinearReg.ipynb) | 怎样用一条直线预测连续数值？梯度下降如何改进这条线？ |
| 2 | [逻辑回归](MachineLearning/LogisticReg.ipynb) | 怎样估计二分类概率？调整决策阈值会改变什么？ |
| 3 | [K-means 聚类](MachineLearning/Kmeans.ipynb) | 没有标签时，如何反复分配样本并更新中心？ |
| 4 | [层次聚类](MachineLearning/CascadedClustering.ipynb) | 如何从单个样本开始逐步合并，并用树状图选择分组？ |
| 5 | [主成分分析 PCA](MachineLearning/PCA.ipynb) | 怎样投影到较少维度，同时尽量保留数据的变化？ |
| 6 | [Q-learning](RLsimple.ipynb) | 智能体如何通过奖励和探索在网格中学会行动？ |

每个笔记本都可以单独学习；线性回归和基础练习适合先学。课中请从上到下逐格运行，遇到问题先作预测、操作演示，再回到代码改变参数或补充实现，并运行紧接着的检查。标为「演示」或「准备」的代码直接运行即可，无需理解或修改 Widget；标为「练习」的代码才是需要关注的算法实现。检查通过只说明满足所给例子，还要观察图形并思考不同输入下的行为。

演示控件的实现集中在 `MachineLearning/teaching_widgets.py`，情景插图位于 `assets/`，都是课程配套文件。请保留整个项目目录结构，不要只复制单个笔记本。修改练习代码后，按笔记本中的提示重新运行相应的训练、演示和检查单元。

## 使用交互图

使用工具栏的运行按钮或 `Shift + Enter` 执行单元格。图上的滑块、播放按钮或其他控件可用于逐步查看计算；先运行其上方的数据和函数单元格，再运行控件单元格。修改了前面的代码后，需要重新运行受影响的后续单元格。若状态混乱，选择 **Kernel → Restart Kernel and Run All Cells**，然后重新操作控件。自动播放有助于看到变化，暂停后逐步查看更适合比较每次更新。

如果控件显示为纯文本或无法响应：

1. 确认是在 JupyterLab 中打开 notebook，而不是仅在 GitHub 或静态预览中查看；静态预览不能执行 Python 或响应控件。
2. 确认当前 notebook 右上角的内核是安装依赖的 Python 环境。在 notebook 中运行 `import sys; print(sys.executable)`，再在终端用该解释器执行 `python -m pip install -r requirements.txt`（将 `python` 替换为显示的解释器路径）。
3. 重启内核并刷新浏览器页面，然后从头顺序运行。若仍看不到控件，在该环境检查 `python -m pip show ipywidgets jupyterlab ipykernel`；必要时重新安装依赖并重新启动 JupyterLab。
4. 图形中的中文若显示为方框，请在系统安装中文字体并重启内核；具体字体取决于操作系统。无中文字体不影响算法数值。

`MachineLearning/Manuscript.md` 是旧课程草稿，包含尚未完成的章节和未经校正的示例；请以实际 notebook 的内容为准。
