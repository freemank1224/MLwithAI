# 课程插图与算法示意图

决策树课程使用 `watermelon_*.png` 的八张西瓜主题示意图，分别展示决策路径、分类划分、随机森林抽样、模型脉络、回归划分、两轮提升与 XGBoost 计算。图中的样本是教学用虚构数据。可在项目根目录运行 `python assets/generate_watermelon_diagrams.py` 重新生成；脚本需要系统中有中文字体。

`watermelon_xgb_gradient.png` 和 `watermelon_xgb_gain.png` 进一步用平方损失曲线与四瓜算例解释梯度、叶权和分裂收益；运行 `python assets/generate_watermelon_xgb_intuition.py` 可重新生成。

较早的 `tree_learning_map.png`、`tree_split_mechanics.png`、`forest_vs_boosting.png`、`gbdt_residuals.png` 与 `xgboost_mechanics.png` 仍保留在项目中，可运行 `python assets/generate_tree_diagrams.py` 重新生成。

七张插图使用 Codex 内置 imagegen 生图工具分别生成。它们用于故事引入；精确的数据和计算由笔记本代码提供。

`tree_decision_diagram.png` 是旧版课程使用的决策树示意图，现保留作历史素材；规则仅用于说明读图方式，不是模型训练结果。

生成提示词：使用简洁的教学流程图，根节点为“广告 ≥ 5？”，是则预测售罄，否则继续判断“有活动？”，是则预测售罄，否则预测未售罄；每个判断节点只有两条标记为“是”和“否”的箭头，不加入人物或装饰背景。使用 Codex 内置 imagegen 工具生成。

## 实际提示词

每张图使用以下共同提示词，加上相应的场景描述：

> Use case: illustration-story. Create a single wide horizontal triptych for the opening of a beginner machine-learning Jupyter lesson. Three clearly separated equal-sized panels left-to-right, readable without words, same characters and objects between panels. Warm inviting editorial storybook illustration for adult learners, clean composition, soft paper texture, limited teal, coral and golden palette, clear shapes, gentle humor. Each panel should be a distinct scene, not a graph or code screenshot. No words, no letters, no numbers, no watermark, no logos.

- `linear_scene.png`：A school teacher wonders how study time relates to test scores. Three sequential storyboard panels: student studies briefly and teacher sees a modest score; another studies longer and has a higher score; teacher looks at a handful of dots and wonders about an unobserved study time. No charts with precise lines, no equations.
- `logistic_scene.png`：A friendly small bakery owner deciding whether to prepare an extra tray of bread. Three sequential storyboard panels: morning with uncertain customer demand; a day with a small neighborhood event and more customers; owner choosing whether to bake an extra tray, outcome uncertain. No probability numbers.
- `tree_scene.png`：A friendly orchard keeper sorting fruit based on simple visible qualities. First panel: the keeper looks at a mixed basket of fruit and wonders how to sort them. Second: the keeper separates fruits into two groups by one visible quality. Third: several small groups of fruit sit in baskets and the keeper compares their sorting choices.
- `kmeans_scene.png`：A friendly neighborhood planner deciding where to place three community service stations. Three sequential storyboard panels: a few families in homes scattered around a simple map; three colored tentative station markers; families gathering around their nearby station. Do not depict a predetermined correct solution.
- `hierarchy_scene.png`：A fruit shop owner organizing a small pile of apples, oranges and bananas. Three sequential storyboard panels: each fruit is separate; two similar fruits are placed together; small groups sit on shelves while the owner considers whether to combine them. No algorithmic dendrogram.
- `pca_scene.png`：A traveler sends a picture over a slow connection. Three sequential storyboard panels: traveler sees a clear handwritten digit on a small screen; the tiny image is squeezed into a compact package; a friend receives a simplified, slightly blurry version of the digit. Clearly show original, compression, reconstruction without text.
- `qlearning_scene.png`：A cheerful small delivery robot in a simple square warehouse. Three sequential storyboard panels: robot at a start square looks toward a parcel destination across shelves; robot tries a blocked way and stops; robot tries a different route around shelves toward the destination. Simple top-down grid, consistent robot and warehouse.

深度学习专题的 `cnn_examples.png` 使用 MNIST 和 Sign Language MNIST 训练子集中的六张原图，放大并加上课堂观察提示。MNIST 图像署名 Yann LeCun、Corinna Cortes，图版沿用 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)。手势照片由 tecperson 发布，使用 CC0。[数据来源与抽样说明](../MachineLearning/data/image_recognition/README.md)

`cnn_window.png` 中的 4 × 5 输入为教学构造，橙框把一个 2 × 2 输入窗口与其特征图上的对应数值联系起来。

第二课的 `augmentation_views.png` 用同一张手势训练图演示位移、曝光变化与指尖移出画面。第三课的 `transfer_routes.png` 用数字训练图和结构图标出迁移时的卷积权重及新分类头。数字图版沿用 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)（MNIST © Yann LeCun、Corinna Cortes），手势图来自 tecperson 发布的 CC0 数据。[数据来源](../MachineLearning/data/image_recognition/README.md)
