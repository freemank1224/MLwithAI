"""生成西瓜课程示意图。运行：python assets/generate_watermelon_diagrams.py。
图中数字均为虚构教学例子，不是教材原始数据。需安装 matplotlib 和中文字体。
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
font = Path('/System/Library/Fonts/STHeiti Medium.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
INK='#173b39'; SUB='#4c6862'; GREEN='#238270'; PALE='#e4f3eb'; RED='#d56d5a'; ROSE='#fcece6'; BLUE='#eaf1f5'; WHITE='#ffffff'; BORDER='#bed3c9'; GOLD='#fcf3d7'

def canvas(title, subtitle):
    fig, ax=plt.subplots(figsize=(14,8),dpi=170)
    fig.patch.set_facecolor('#f7faf6'); ax.set_facecolor('#f7faf6')
    ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis('off')
    ax.text(.4,7.62,title,fontsize=21,color=INK,va='center',weight='bold')
    ax.text(.4,7.18,subtitle,fontsize=12,color=SUB,va='center')
    return fig,ax

def box(ax,x,y,w,h,text,face=WHITE,edge=BORDER,fs=13,weight='normal'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.035,rounding_size=.12',facecolor=face,edgecolor=edge,linewidth=1.4))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,weight=weight,linespacing=1.55)

def arrow(ax,start,end,color=GREEN):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,linewidth=2,color=color,shrinkA=2,shrinkB=2))

def text(ax,x,y,s,fs=11,color=SUB):
    ax.text(x,y,s,fontsize=fs,color=color,ha='center',va='center',linespacing=1.4)

def save(fig,name):
    fig.savefig(OUT/name,dpi=170,bbox_inches='tight',pad_inches=.15,facecolor=fig.get_facecolor()); plt.close(fig)

# Reading a path in an intentionally invented tree
f,a=canvas('挑西瓜：沿一条路走到结果','示意规则，不代表实际训练出来的树；读图时每一步只选择一个答案。')
box(a,4.9,5.7,4.2,.82,'纹理清晰吗？',PALE,GREEN,17,'bold')
box(a,1.0,3.5,4.0,.82,'根蒂蜷缩吗？',PALE,GREEN,17)
box(a,9.0,3.5,4.0,.82,'预测：坏瓜',ROSE,RED,17,'bold')
arrow(a,(5.5,5.68),(3.1,4.36)); arrow(a,(8.5,5.68),(10.9,4.36))
text(a,3.7,5.0,'是',14,GREEN); text(a,10.25,5.0,'否',14,RED)
box(a,.35,1.15,3.1,.82,'预测：好瓜',PALE,GREEN,17,'bold')
box(a,4.5,1.15,3.1,.82,'预测：坏瓜',ROSE,RED,17,'bold')
arrow(a,(2.2,3.47),(1.9,2.02)); arrow(a,(3.8,3.47),(6.0,2.02))
text(a,1.72,2.72,'是',14,GREEN); text(a,5.55,2.72,'否',14,RED)
box(a,8.6,1.0,4.7,1.05,'例：纹理清晰、根蒂蜷缩\n沿“是 → 是”到达好瓜',WHITE,BORDER,13)
save(f,'watermelon_tree_path.png')

# Classification split, explicit six samples
f,a=canvas('选哪个问题，才能把好瓜和坏瓜分得更清楚？','以下 6 个瓜是虚构的演示数据；先比较候选问题，再在各组重复。')
box(a,.5,5.82,13,.75,'西瓜： A   B   C   D   E   F     |     答案： 好   好   坏   好   坏   坏',WHITE,BORDER,14)
text(a,7,5.45,'原来：3 个好瓜、3 个坏瓜，基尼不纯度 = 0.5',12)
box(a,.6,3.82,6.05,1.23,'问题 A：色泽是青绿还是乌黑？\n青绿：好、好、坏    乌黑：好、坏、坏\n两组不纯度各为 4/9',BLUE,'#9dbfc9',13)
box(a,7.35,3.82,6.05,1.23,'问题 B：纹理是清晰还是模糊？\n清晰：好、好、坏、好    模糊：坏、坏\n不纯度分别为 0.375 和 0',PALE,GREEN,13)
box(a,.6,2.73,6.05,.68,'加权： 3/6 × 4/9 + 3/6 × 4/9 = 4/9',WHITE,BORDER,13)
box(a,7.35,2.73,6.05,.68,'加权： 4/6 × 0.375 + 2/6 × 0 = 0.25',WHITE,GREEN,13,'bold')
text(a,7,2.26,'先选 B：整体更纯；“纹理清晰”那组仍可继续提问。',13,GREEN)
box(a,3.8,1.2,6.4,.65,'纹理清晰？ → 是：继续问；否：预测坏瓜',PALE,GREEN,14)
text(a,7,.55,'比较同一个节点的候选划分：基尼越低，划分后越纯。',11)
save(f,'watermelon_split_mechanics.png')

# Concrete bootstrap and averaging
f,a=canvas('随机森林：同一批瓜，抽出不同的训练集','四个字母代表四个虚构样本；每棵树各自有放回抽 4 次。')
box(a,4.7,5.85,4.6,.64,'原始训练集： A   B   C   D',WHITE,BORDER,14)
for x,seq,feat,prob in [(0.45,'A  A  B  C','纹理、色泽','0.75'),(4.85,'A  C  C  D','根蒂、敲声','0.5'),(9.25,'B  B  C  D','纹理、根蒂','0.25')]:
    box(a,x,4.1,4.0,.95,'抽样： '+seq+'\n同一个瓜可重复，也可能没抽到',BLUE,'#9dbfc9',12)
    box(a,x,2.75,4.0,.85,'某次分裂候选特征：\n'+feat,PALE,GREEN,12)
    box(a,x,1.55,4.0,.66,'同一新瓜预测好瓜概率：'+prob,WHITE,BORDER,12)
    arrow(a,(7,5.82),(x+2,5.1)); arrow(a,(x+2,4.07),(x+2,3.65)); arrow(a,(x+2,2.72),(x+2,2.25))
text(a,7,.75,'三个假设概率取平均：(0.75 + 0.5 + 0.25) / 3 = 0.5；最终结果并非某一棵树单独决定。',12,GREEN)
save(f,'watermelon_forest_bootstrap.png')

# Parallel vs sequential
f,a=canvas('同样是多棵树，森林与提升怎样合作？','左边的树分别学习；右边后一棵树要先知道已有预测。')
box(a,.5,5.8,6.2,.7,'随机森林：分别训练，再聚合',PALE,GREEN,16,'bold')
box(a,7.3,5.8,6.2,.7,'梯度提升：一棵接一棵修正',ROSE,RED,16,'bold')
box(a,2.05,4.75,3.1,.6,'同一批训练西瓜',WHITE,BORDER,13)
for x in (1.55,3.6,5.65):
    arrow(a,(3.6,4.72),(x,4.12)); box(a,x-.7,3.35,1.4,.65,'树',BLUE,'#9dbfc9',14)
    arrow(a,(x,3.33),(3.6,2.5))
box(a,1.2,1.65,4.8,.75,'好瓜分类：聚合预测概率\n甜度回归：平均预测值',PALE,GREEN,12)
box(a,7.48,4.7,1.8,.67,'初始甜度',WHITE,BORDER,12)
box(a,9.76,4.7,1.65,.67,'第一棵树',ROSE,RED,12)
box(a,11.9,4.7,1.65,.67,'第二棵树',ROSE,RED,12)
arrow(a,(9.29,5.03),(9.73,5.03),RED); arrow(a,(11.43,5.03),(11.87,5.03),RED)
text(a,10.6,4.22,'重新计算残差，再训练下一棵树',12)
arrow(a,(12.7,4.67),(12.7,3.7),RED)
box(a,8.0,2.72,5.2,.82,'甜度预测 = 初值 + 第一轮修正\n+ 第二轮修正 + ……',ROSE,RED,13)
text(a,7,.7,'两者都要留意过拟合：多棵树并不保证一定预测准确。',12)
save(f,'watermelon_forest_vs_boosting.png')

# Family tree
f,a=canvas('学完森林，再看另一条路','这张地图展示模型之间的联系；箭头不是说随机森林会变成 GBDT。')
box(a,5.0,5.7,4,.8,'决策树：一连串问题',PALE,GREEN,16,'bold')
box(a,.8,3.75,4.8,.95,'Bagging：各树独立学习\n最后聚合结果',BLUE,'#9dbfc9',14)
box(a,8.4,3.75,4.8,.95,'Boosting：后一轮依据\n已有预测继续修正',ROSE,RED,14)
arrow(a,(5.6,5.67),(3.2,4.73)); arrow(a,(8.4,5.67),(10.8,4.73),RED)
box(a,.8,1.8,4.8,.9,'随机森林：Bagging\n+ 分裂时随机选择候选特征',PALE,GREEN,13)
box(a,8.4,1.8,4.8,.9,'GBDT：新树根据损失的\n负梯度逐轮修正',ROSE,RED,13)
arrow(a,(3.2,3.72),(3.2,2.73)); arrow(a,(10.8,3.72),(10.8,2.73),RED)
box(a,8.4,.3,4.8,.75,'XGBoost 树模型：一种梯度提升实现',GOLD,'#c5a655',13)
arrow(a,(10.8,1.77),(10.8,1.08),'#b68c39')
save(f,'watermelon_learning_map.png')

# regression split
f,a=canvas('回归树如何选择划分？','四个瓜的密度与甜度是虚构数据；回归树以一组瓜的平均甜度作为预测。')
box(a,.5,5.85,13,.67,'密度： 0.2     0.4     0.6     0.8      |      甜度： 6     8     10     12',WHITE,BORDER,14)
box(a,.65,3.65,6,1.6,'候选 A：密度 ≤ 0.3\n左边 6 → 均值 6，平方误差和 0\n右边 8、10、12 → 均值 10，平方误差和 8\n总体均方误差 = (0 + 8) / 4 = 2',BLUE,'#9dbfc9',12)
box(a,7.35,3.65,6,1.6,'候选 B：密度 ≤ 0.5\n左边 6、8 → 均值 7，平方误差和 2\n右边 10、12 → 均值 11，平方误差和 2\n总体均方误差 = (2 + 2) / 4 = 1',PALE,GREEN,12)
text(a,7,3.1,'比较每组误差之和后，B 的误差更小。',14,GREEN)
box(a,2.5,1.35,9,1.0,'分类树：比较基尼等类别混杂程度，叶子给出类别\n回归树：比较数值误差，叶子给出一组样本的平均值',WHITE,BORDER,13)
save(f,'watermelon_regression_split.png')

# two boosting rounds
f,a=canvas('梯度提升：让两棵小树接力改甜度预测','三颗瓜数据为虚构，使用平方误差；密度是输入，甜度是预测目标。')
box(a,.35,5.82,4.05,.75,'密度：0.2、0.5、0.8\n真实甜度：6、8、10',WHITE,BORDER,12)
box(a,4.95,5.82,4.05,.75,'初始预测：8、8、8\n残差：−2、0、2',BLUE,'#9dbfc9',12)
box(a,9.55,5.82,4.05,.75,'学习率 η = 0.5\n每次只加一半修正',GOLD,'#c5a655',12)
arrow(a,(4.42,6.2),(4.93,6.2)); arrow(a,(9.02,6.2),(9.53,6.2))
box(a,.5,3.98,6.2,1.15,'第 1 棵树：按密度 ≤ 0.35 划分\n拟合残差，输出 −2、1、1\n预测 = (8, 8, 8) + 0.5 × (−2, 1, 1)',PALE,GREEN,12)
box(a,7.25,3.98,6.2,1.15,'第 1 轮后预测：7、8.5、8.5\n重新计算残差 = 真实 − 预测\n得到 −1、−0.5、1.5',WHITE,BORDER,12)
arrow(a,(6.72,4.55),(7.22,4.55))
box(a,.5,1.9,6.2,1.15,'第 2 棵树：按密度 ≤ 0.65 划分\n拟合新残差，输出 −0.75、−0.75、1.5\n预测再加 0.5 × 树的输出',ROSE,RED,12)
box(a,7.25,1.9,6.2,1.15,'第 2 轮后预测：6.625、8.125、9.25\n训练均方误差约 0.323\n初始约 2.667 → 第 1 轮约 1.167',WHITE,BORDER,12)
arrow(a,(10.35,3.94),(3.7,3.08),RED); arrow(a,(6.72,2.47),(7.22,2.47),RED)
text(a,7,.75,'每轮先用当前预测计算残差，再让新树学这些残差；不是拿前一棵树的预测直接当最终结果。',11)
save(f,'watermelon_gbdt_rounds.png')

# Numerical xgboost
f,a=canvas('XGBoost：让数字一步步决定修正量','虚构 4 瓜，平方损失 ½(预测−真实)²；只展示 L2 正则，λ=1，γ=0。')
box(a,.45,5.82,13.1,.75,'真实甜度 y： 6，8，10，12     |     当前预测： 9，9，9，9',WHITE,BORDER,14)
box(a,.45,4.18,4.05,1.15,'每个瓜先计算：\ng = 预测 − 真实 = 3，1，−1，−3\nh = 1，1，1，1',BLUE,'#9dbfc9',12)
box(a,4.95,4.18,4.05,1.15,'候选分组：前两个 / 后两个\n左：G = 4，H = 2\n右：G = −4，H = 2',PALE,GREEN,12)
box(a,9.45,4.18,4.05,1.15,'先估计两片叶的修正\nw = −G / (H + λ)\n左：−4/3；右：4/3',GOLD,'#c5a655',12)
arrow(a,(4.52,4.75),(4.92,4.75)); arrow(a,(9.02,4.75),(9.42,4.75))
box(a,.75,2.05,12.5,1.5,'父节点：G = 0，H = 4\n分裂增益 = ½ × [4²/(2+1) + (−4)²/(2+1) − 0²/(4+1)] − 0\n= 16/3；收益为正，此简化例子支持分裂',WHITE,BORDER,13)
text(a,7,1.35,'g 指示损失如何随预测变化；h 描述变化速度。λ 使叶权更保守，γ 是新增叶的代价。',11)
text(a,7,.65,'示意公式省略 L1 与其他约束；二阶近似和正则化并不是 XGBoost 独有。',11)
save(f,'watermelon_xgboost_mechanics.png')
print('Generated 8 watermelon diagrams in',OUT)
