"""Generate the decision-tree lesson diagrams with matplotlib.

Run from the repository root: python assets/generate_tree_diagrams.py
A CJK font is required. Adjust FONT_PATH for the local system if needed.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent
FONT_PATH = Path('/System/Library/Fonts/STHeiti Medium.ttc')
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'
INK = '#203844'; MUTED = '#526b75'; TEAL = '#167d7f'; LIGHT = '#e5f4f1'; CORAL = '#db765f'; PALE = '#fff0e8'; BLUE = '#eaf1fa'; GOLD = '#f7edcf'; WHITE = '#ffffff'; BORDER = '#cbdad9'


def canvas(title, subtitle='', width=14, height=7.7):
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    fig.patch.set_facecolor('#f8faf8')
    ax.set_facecolor('#f8faf8'); ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')
    ax.text(.45, 7.68, title, fontsize=22, color=INK, weight='bold', va='center')
    if subtitle: ax.text(.46, 7.24, subtitle, fontsize=12, color=MUTED, va='center')
    return fig, ax


def box(ax, x, y, w, h, text, face=WHITE, edge=BORDER, fs=13, color=INK, bold=False, align='center'):
    p = FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.035,rounding_size=0.14', linewidth=1.4, edgecolor=edge, facecolor=face)
    ax.add_patch(p)
    ax.text(x+w/2 if align=='center' else x+.18, y+h/2, text, ha=align, va='center', fontsize=fs, color=color, weight='bold' if bold else 'normal', linespacing=1.48)
    return p


def arrow(ax, a, b, color=TEAL, lw=2, style='-|>', ms=14):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=ms,linewidth=lw,color=color,shrinkA=2,shrinkB=2))


def label(ax, x, y, s, fs=11, color=MUTED, ha='center'):
    ax.text(x,y,s,fontsize=fs,color=color,ha=ha,va='center',linespacing=1.35)


def save(fig, name):
    fig.savefig(OUT/name, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=.16)
    plt.close(fig)


# 1. A conceptual map: arrows represent composition and specialization.
f,a=canvas('从一棵树到一组树', '先看如何划分，再看多棵树如何合作；XGBoost 的树模型是梯度提升树的一种实现。')
box(a,5.25,5.65,3.5,.9,'决策树\n逐节点提问 → 到叶节点预测',LIGHT,TEAL,15,bold=True)
box(a,1.0,3.72,4.2,1.08,'树的 Bagging：并行训练\n不同重采样数据 → 聚合预测',BLUE,'#8bb0ca',14)
box(a,8.8,3.72,4.2,1.08,'Boosting：按顺序加入模型\n新模型修正已有模型',PALE,CORAL,14)
arrow(a,(6.15,5.58),(3.5,4.85)); arrow(a,(7.85,5.58),(10.65,4.85),CORAL)
box(a,1.0,1.75,4.2,1.08,'随机森林\nBagging + 分裂时随机选特征',LIGHT,TEAL,14,bold=True)
box(a,5.55,1.75,2.35,1.08,'AdaBoost\n另一类提升方法',WHITE,BORDER,12)
arrow(a,(9.0,3.68),(7.55,2.9),CORAL)
box(a,8.8,1.75,4.2,1.08,'梯度提升树（GBDT）\n树依次拟合损失的负梯度',PALE,CORAL,14,bold=True)
arrow(a,(3.1,3.66),(3.1,2.9)); arrow(a,(10.9,3.66),(10.9,2.9),CORAL)
box(a,8.8,.25,4.2,.83,'XGBoost 树模型：梯度提升实现',GOLD,'#cfaa55',14,bold=True)
arrow(a,(10.9,1.7),(10.9,1.14),'#b68c39')
label(a,3.1,.9,'分类可聚合概率或投票；回归取平均',11)
save(f,'tree_learning_map.png')

# 2. Root split is calculated on an explicit tiny dataset.
f,a=canvas('决策树如何选择一个问题？','先在当前节点比较候选划分；选好后，左右子节点再各自重复。')
box(a,.45,5.67,13.1,.85,'训练样本按广告投入排序：  1   2   3   4   5   6     |     标签：  未   未   未   售   售   售',WHITE,BORDER,14)
label(a,7,5.36,'当前节点：3 个未售罄、3 个售罄  →  基尼不纯度 G = 1 − (3/6)² − (3/6)² = 0.5',12)
box(a,.65,3.7,5.65,1.22,'候选 A：广告 ≤ 2.5？\n左：未、未   G = 0\n右：未、售、售、售   G = 0.375',BLUE,'#a6bdd0',13)
box(a,7.7,3.7,5.65,1.22,'候选 B：广告 ≤ 3.5？\n左：未、未、未   G = 0\n右：售、售、售   G = 0',LIGHT,TEAL,13)
box(a,.65,2.64,5.65,.65,'加权后 = (2/6) × 0 + (4/6) × 0.375 = 0.25',WHITE,BORDER,12)
box(a,7.7,2.64,5.65,.65,'加权后 = (3/6) × 0 + (3/6) × 0 = 0',WHITE,TEAL,12,bold=True)
label(a,7,2.24,'B 的加权不纯度更低；此例选择 B。',13,TEAL)
box(a,4.65,1.22,4.7,.67,'广告 ≤ 3.5？',LIGHT,TEAL,14,bold=True)
arrow(a,(5.5,1.2),(3.7,.77)); arrow(a,(8.5,1.2),(10.3,.77))
box(a,1.95,.16,3.5,.57,'是 → 叶节点：预测未售罄',BLUE,'#a6bdd0',12)
box(a,8.55,.16,3.5,.57,'否 → 叶节点：预测售罄',PALE,CORAL,12)
save(f,'tree_split_mechanics.png')

# 3. Side-by-side comparison, with separate prediction aggregations.
f,a=canvas('随机森林与梯度提升树：两种合作方式','两者都由树组成，但训练顺序、树之间的关系和组合方式不同。')
box(a,.5,5.8,6.2,.72,'随机森林：并行构建，再聚合',LIGHT,TEAL,16,bold=True)
box(a,7.3,5.8,6.2,.72,'GBDT：顺序构建，逐步修正',PALE,CORAL,16,bold=True)
box(a,2.05,4.8,3.1,.58,'同一训练集',WHITE,BORDER,13)
for x in (1.55,3.6,5.65):
    arrow(a,(3.6,4.78),(x,4.12)); box(a,x-.75,3.35,1.5,.69,'树',BLUE,'#91b1c7',14,bold=True)
label(a,3.6,1.18,'各自有放回抽样；节点处随机选候选特征',10)
for x in (1.55,3.6,5.65): arrow(a,(x,3.32),(3.6,2.52))
box(a,1.47,1.68,4.26,.72,'聚合预测：分类聚合概率 / 投票\n回归取平均',LIGHT,TEAL,12)
box(a,7.72,4.74,1.62,.66,'初始预测',WHITE,BORDER,12)
box(a,10.02,4.74,1.52,.66,'第 1 棵树',PALE,CORAL,12)
box(a,12.0,4.74,1.5,.66,'第 2 棵树',PALE,CORAL,12)
arrow(a,(9.35,5.07),(10,5.07),CORAL); arrow(a,(11.55,5.07),(11.98,5.07),CORAL)
label(a,10.4,4.35,'根据当前误差 / 负梯度训练下一棵树',11)
arrow(a,(12.75,4.68),(12.75,3.85),CORAL)
box(a,8.25,2.75,4.95,1.0,'最终预测 = 初始值 + 每棵树的修正\n（修正通常乘学习率）',PALE,CORAL,13)
label(a,7,.64,'森林：降低单棵树的波动   ·   提升：逐轮优化损失；两者都需要控制过拟合',12)
save(f,'forest_vs_boosting.png')

# 4. A numeric toy example; regression stumps and squared error.
f,a=canvas('梯度提升回归：用残差作下一步方向','平方误差回归的特例：新树拟合 y − 当前预测；分类等其他损失要用对应负梯度。')
box(a,.45,5.7,3.1,.82,'样本 x： 1，2，3\n真实销量 y： 2，4，6',WHITE,BORDER,13)
box(a,4.08,5.7,3.15,.82,'初始预测：平均值 4\n当前预测： 4，4，4',BLUE,'#91b1c7',13)
box(a,7.78,5.7,2.85,.82,'残差 y − 预测\n−2，0，2',PALE,CORAL,13)
box(a,11.15,5.7,2.4,.82,'拟合残差\n训练一棵树',LIGHT,TEAL,13)
for x1,x2 in ((3.57,4.05),(7.25,7.75),(10.65,11.12)): arrow(a,(x1,6.1),(x2,6.1))
box(a,.7,3.65,5.55,1.18,'示例树桩：x ≤ 1.5？\n左叶输出 −2；右叶输出 (0 + 2) / 2 = 1\n因此树的输出 h(x)： −2，1，1',LIGHT,TEAL,13)
box(a,7.1,3.65,6.1,1.18,'学习率 η = 0.5\n新预测 = 原预测 + η × 树输出\n= (4，4，4) + 0.5 × (−2，1，1)',PALE,CORAL,13)
arrow(a,(6.3,4.24),(7.05,4.24),CORAL)
box(a,1.2,1.55,5.3,1.0,'更新后预测：3，4.5，4.5\n再计算残差，继续训练下一棵树',WHITE,BORDER,14,bold=True)
box(a,7.45,1.55,5.3,1.0,'训练均方误差：\n8/3 ≈ 2.67  →  3.5/3 ≈ 1.17',WHITE,BORDER,14,bold=True)
label(a,7,.6,'示例只演示一次更新；训练误差下降不保证测试误差每轮都下降。',12)
save(f,'gbdt_residuals.png')

# 5. XGBoost mechanism with stated objective conventions and constrained scope.
f,a=canvas('XGBoost：怎样给新树打分？','以 gbtree 和 L2 正则为例；二阶近似等方法并非 XGBoost 独有。')
box(a,.5,5.55,3.0,1.0,'损失对预测分数求导\n得到一阶梯度 g_i\n和二阶导数 h_i',BLUE,'#91b1c7',13)
box(a,4.15,5.55,3.1,1.0,'把样本分到叶节点\n叶内求和：G = Σg_i\nH = Σh_i',LIGHT,TEAL,13)
box(a,8.05,5.55,5.4,1.0,'叶子最优权重\nw* = −G / (H + λ)\nλ 控制叶权的 L2 惩罚',GOLD,'#cfaa55',14,bold=True)
arrow(a,(3.53,6.05),(4.12,6.05)); arrow(a,(7.28,6.05),(8.02,6.05))
box(a,.65,3.15,12.7,1.55,'候选分裂的增益（左 L、右 R；父节点 G = G_L + G_R，H = H_L + H_R）\n½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ\nγ 是新增叶节点的复杂度代价；比较增益，决定是否划分。',WHITE,BORDER,14)
arrow(a,(10.75,5.5),(10.75,4.76),'#b68c39')
box(a,.8,1.32,3.85,1.05,'目标函数：拟合损失 + 正则项\n既追求拟合，也控制复杂度',BLUE,'#91b1c7',12)
box(a,5.05,1.32,3.85,1.05,'拟合新树后加入原模型\n通常再乘学习率 η',LIGHT,TEAL,12)
box(a,9.3,1.32,3.85,1.05,'还能使用采样、缺失值处理等\n细节取决于实现与参数',PALE,CORAL,12)
label(a,7,.55,'图示省略 L1 正则项及其他约束；g、h 来自所选损失函数。',11)
save(f,'xgboost_mechanics.png')
print('Generated five decision-tree diagrams in', OUT)
