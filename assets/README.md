# 课程插图与算法示意图

决策树课程另有五张算法图：`tree_learning_map.png` 展示算法关系，`tree_split_mechanics.png` 展示划分的计算，`forest_vs_boosting.png` 比较组合方式，`gbdt_residuals.png` 展示残差更新，`xgboost_mechanics.png` 解释叶权和分裂增益。图中的数据是独立的教学示例。可在项目根目录运行 `python assets/generate_tree_diagrams.py` 重新生成；脚本需要系统中有中文字体。

七张插图使用 Codex 内置 imagegen 生图工具分别生成。它们用于故事引入；精确的数据和计算由笔记本代码提供。

`tree_decision_diagram.png` 是另外生成的决策树示意图，取代水果情景图作为课程引入。它展示条件判断、分支与预测叶节点，规则仅用于说明读图方式，不是模型训练结果。

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
