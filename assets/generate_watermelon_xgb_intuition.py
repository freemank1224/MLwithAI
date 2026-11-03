"""为西瓜课程生成 XGBoost 数值直觉图。需要 matplotlib 和中文字体。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
FONT = Path('/System/Library/Fonts/STHeiti Medium.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
else:
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
BG = '#f7faf6'; INK = '#173b39'; SUB = '#4c6862'; GREEN = '#238270'; PALE = '#e4f3eb'; RED = '#d56d5a'; ROSE = '#fcece6'; BLUE = '#eaf1f5'; BORDER = '#bed3c9'; WHITE = '#fff'; GOLD = '#fcf3d7'


def box(ax, x, y, w, h, lines, face=WHITE, edge=BORDER, size=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.025,rounding_size=.08', facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(x+w/2, y+h/2, lines, ha='center', va='center', fontsize=size, color=INK, linespacing=1.55)


def arrow(ax, start, end, color=GREEN):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=16,linewidth=2,color=color))


def title(fig, t, sub):
    fig.text(.045,.95,t,fontsize=24,color=INK,weight='bold',va='top')
    fig.text(.045,.89,sub,fontsize=13,color=SUB,va='top')


def write(fig, path):
    fig.savefig(OUT/path,dpi=180,facecolor=BG,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)


# Half squared loss: exact derivatives and a local tangent, shown beside four samples.
f = plt.figure(figsize=(14,8),dpi=180,facecolor=BG)
title(f,'先看一颗瓜：梯度在告诉我们什么？','真实甜度是 6，当前预测是 9；采用损失 L = ½ ×（预测 − 真实）²。')
a=f.add_axes([.08,.18,.48,.61],facecolor='white')
x=np.linspace(4.4,10.3,250); y=.5*(x-6)**2
assert .5*(9-6)**2 == 4.5 and .5*(8-6)**2 == 2
a.plot(x,y,color=GREEN,lw=3); a.scatter([6,8,9],[0,2,4.5],color=[GREEN,GREEN,RED],s=65,zorder=4)
a.plot([8.3,9.5],[4.5+3*(8.3-9),4.5+3*(9.5-9)],color=RED,ls='--',lw=2)
a.annotate('当前：预测 9\n损失 4.5',xy=(9,4.5),xytext=(9.18,3.7),fontsize=12,color=INK)
a.annotate('往左改为 8\n损失降到 2',xy=(8,2),xytext=(7.0,4.2),arrowprops={'arrowstyle':'->','color':GREEN,'lw':1.8},fontsize=12,color=INK)
a.text(5.0,6.9,'红色虚线：当前位置的切线\n斜率 g = 9 − 6 = 3',fontsize=12,color=RED,linespacing=1.45)
a.set(xlim=(4.4,10.4),ylim=(0,9),xlabel='预测甜度',ylabel='损失 L');a.grid(alpha=.2)
b=f.add_axes([.61,.16,.35,.66]);b.set(xlim=(0,1),ylim=(0,1));b.axis('off')
box(b,.02,.68,.96,.25,'g = 预测 − 真实\n9 时 g = 3；8 时 g = 2',PALE,GREEN,14)
box(b,.02,.37,.96,.25,'预测增加 1，g 也增加 1\n这个例子里的二阶信息 h = 1',BLUE,BORDER,14)
box(b,.02,.06,.96,.25,'g > 0：小幅向下修正更有利\ng < 0：小幅向上修正更有利',GOLD,'#c5a655',14)
f.text(.07,.065,'这是平方损失下的示例：其他损失的梯度、二阶信息可能不同。',fontsize=12,color=SUB)
write(f,'watermelon_xgb_gradient.png')

# Candidate gain and gamma for a single binary split under L2-only objective.
f = plt.figure(figsize=(14,8),dpi=180,facecolor=BG)
title(f,'再看四颗瓜：分裂究竟值不值？','虚构甜度 6、8、10、12；当前预测全是 9；只考虑 L2 叶权正则，λ = 1。')
a=f.add_axes([.04,.09,.92,.74]);a.set(xlim=(0,14),ylim=(0,8));a.axis('off')
box(a,3.2,6.72,7.6,.9,'四瓜的 g： 3、1、−1、−3     每瓜 h：1\n父节点：G = 0，H = 4',WHITE,BORDER,14)
box(a,.15,4.45,6.0,1.4,'左边两瓜：G = 3 + 1 = 4，H = 2\n叶修正量 = −4 / (2 + 1) = −4/3',PALE,GREEN,14)
box(a,7.85,4.45,6.0,1.4,'右边两瓜：G = −1 − 3 = −4，H = 2\n叶修正量 = 4 / (2 + 1) = 4/3',ROSE,RED,14)
arrow(a,(5.1,6.68),(3.5,5.9));arrow(a,(8.9,6.68),(10.5,5.9),RED)
box(a,.6,2.45,12.8,1.35,'分裂前：父节点 G = 0，预测修正为 0，结构得分项 = 0\n分裂后：左项 4² / (2 + 1) = 16/3；右项 (−4)² / (2 + 1) = 16/3\n未扣新叶代价时的收益 = ½ × (左项 + 右项 − 父项) = 16/3',BLUE,BORDER,13)
box(a,.6,.5,6.05,1.15,'γ = 0：收益 16/3 > 0\n此例值得分裂',PALE,GREEN,14)
box(a,7.35,.5,6.05,1.15,'γ = 6：收益 16/3 − 6 < 0\n此例不值得新增叶子',GOLD,'#c5a655',14)
f.text(.075,.035,'γ 是新增一片叶子的代价；实际建树还要满足最小叶权等其他约束。',fontsize=12,color=SUB)
write(f,'watermelon_xgb_gain.png')
print('已生成 watermelon_xgb_gradient.png 和 watermelon_xgb_gain.png')
