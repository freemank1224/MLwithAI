"""西瓜课的互动实验：手动组森林与逐轮提升。

在 notebook 中运行 `display(forest_lab())` 或 `display(boosting_lab())`。
实验数据是教学模拟，不代表实际西瓜品质。
"""
from html import escape
from math import isfinite

import ipywidgets as widgets


# 与课程中比较色泽与纹理的六瓜例子一致：1 是好瓜，0 是坏瓜。
MELONS = {
    'A': {'color': '青绿', 'texture': '清晰', 'good': 1},
    'B': {'color': '青绿', 'texture': '清晰', 'good': 1},
    'C': {'color': '青绿', 'texture': '清晰', 'good': 0},
    'D': {'color': '乌黑', 'texture': '清晰', 'good': 1},
    'E': {'color': '乌黑', 'texture': '模糊', 'good': 0},
    'F': {'color': '乌黑', 'texture': '模糊', 'good': 0},
}
FEATURES = {'纹理清晰吗？': ('texture', '清晰'), '色泽青绿吗？': ('color', '青绿')}
NEW_MELON = {'color': '乌黑', 'texture': '清晰'}


def _label_probability(p):
    """让平票保持可见，避免把概念演示的平票误说成某个库的规则。"""
    if p > .5:
        return '好瓜'
    if p < .5:
        return '坏瓜'
    return '持平'


def forest_stump(sample_ids, question):
    """按重复抽取的样本训练一个手选问题的树桩，并预测固定的新瓜。

    节点没法分开时保留原叶，用该节点的样本标签比例；树叶中的重复
    样本按其抽中次数计算。question 是 FEATURES 中的中文问题。
    """
    sample_ids = tuple(sample_ids)
    if len(sample_ids) != 4 or any(x not in MELONS for x in sample_ids):
        raise ValueError('每棵树需要从 A–F 中有放回抽取四次。')
    if question not in FEATURES:
        raise ValueError('请选一个提供的问题。')
    feature, wanted = FEATURES[question]
    yes = [name for name in sample_ids if MELONS[name][feature] == wanted]
    no = [name for name in sample_ids if MELONS[name][feature] != wanted]
    new_goes_yes = NEW_MELON[feature] == wanted
    # 若不能形成两个非空分支，就没有有效分裂，仍在根叶预测。
    did_split = bool(yes and no)
    leaf = (yes if new_goes_yes else no) if did_split else list(sample_ids)
    positives = sum(MELONS[name]['good'] for name in leaf)
    probability = positives / len(leaf)
    return {'sample': sample_ids, 'question': question, 'yes': tuple(yes),
            'no': tuple(no), 'did_split': did_split, 'new_goes_yes': new_goes_yes,
            'leaf': tuple(leaf), 'good_count': positives, 'probability': probability,
            'vote': _label_probability(probability)}


class ForestState:
    """保留最多三棵手动构造的树桩及真实计算出的叶概率。"""
    def __init__(self):
        self.trees = []

    def add_tree(self, sample_ids, question):
        if len(self.trees) >= 3:
            raise ValueError('已经有三棵树；请先重置。')
        tree = forest_stump(sample_ids, question)
        self.trees.append(tree)
        return tree

    def reset(self):
        self.trees.clear()

    def mean_probability(self):
        return sum(t['probability'] for t in self.trees) / len(self.trees) if self.trees else None


def _table(headers, rows):
    """A plain HTML table keeps the notebook readable without extra assets."""
    def cell(x):
        return escape(str(x))
    head = ''.join('<th style="padding:4px 9px;text-align:left">' + cell(x) + '</th>' for x in headers)
    body = ''.join('<tr>' + ''.join('<td style="padding:4px 9px;border-top:1px solid #ddd">' + cell(x) + '</td>' for x in row) + '</tr>' for row in rows)
    return '<table style="border-collapse:collapse;margin:6px 0"><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table>'


def forest_lab():
    """构造森林互动控件；返回 VBox，可多次创建而互不影响。"""
    state = ForestState()
    intro = widgets.HTML(value=(
        '<h3>动手组一片小森林</h3>'
        '<p>目标：判断一颗<strong>乌黑、纹理清晰</strong>的新瓜是否为好瓜。'
        '先给每棵树从 A–F 中<strong>有放回抽四次</strong>（因此可以重复），再选一个问题，'
        '点“种一棵树”；重复三次，看看不同树如何合并。以下六瓜与前面的教学小例子相同。</p>'
        + _table(['瓜', '色泽', '纹理', '真实标签'],
                 [(name, m['color'], m['texture'], '好瓜' if m['good'] else '坏瓜') for name, m in MELONS.items()])
        + '<p>这里让你亲手选问题来观察影响，每棵树只问一次。真正随机森林通常每个节点随机抽候选特征，'
          '再从候选中选择较好的划分；这里是简化实验。</p>'))
    draws = [widgets.Dropdown(options=list(MELONS), value=v, description=f'第{i+1}次',
                              layout=widgets.Layout(width='150px'), style={'description_width': '55px'})
             for i, v in enumerate(('A', 'B', 'C', 'E'))]
    question = widgets.Dropdown(options=list(FEATURES), value='纹理清晰吗？', description='问题',
                                layout=widgets.Layout(width='270px'), style={'description_width': '45px'})
    add = widgets.Button(description='种一棵树', button_style='primary', tooltip='用当前抽样建一棵树')
    reset = widgets.Button(description='重新开始', tooltip='清空三棵树')
    status = widgets.HTML(value='')
    result = widgets.HTML(value='')

    def render():
        if not state.trees:
            result.value = '<p>还没有树：先选择四次抽样与一个问题。</p>'
            return
        rows = []
        for i, tree in enumerate(state.trees, 1):
            sample = ' '.join(tree['sample'])
            if tree['did_split']:
                leaf = ('是' if tree['new_goes_yes'] else '否') + ' → ' + ' '.join(tree['leaf'])
                note = f"叶中 {tree['good_count']} / {len(tree['leaf'])} 为好瓜"
            else:
                leaf = '无法分成两组 → 根叶'
                note = f"根叶中 {tree['good_count']} / {len(tree['leaf'])} 为好瓜"
            rows.append((f'树 {i}', sample, tree['question'], leaf, note,
                         f"{tree['probability']:.3f}", tree['vote']))
        mean = state.mean_probability()
        votes = [tree['vote'] for tree in state.trees]
        result.value = (_table(['树', '抽样', '问题', '新瓜到达', '概率怎么算', '好瓜概率', '单树判断'], rows)
                        + f"<p><strong>森林平均好瓜概率：{mean:.3f}</strong>；"
                          f"各树判断：{'、'.join(votes)}。概率平均与硬投票不一定相同。</p>"
                        + '<p>本演示将单棵树概率恰为 0.5 标为“持平”；实际库的类别平票处理可能不同。'
                          '如果某次抽样无法分成两组，该树保留根叶，用全部抽样的好瓜比例预测。</p>')

    def on_add(_):
        try:
            state.add_tree([d.value for d in draws], question.value)
        except ValueError as exc:
            status.value = '<p>' + escape(str(exc)) + '</p>'
            return
        status.value = f'<p>已建 {len(state.trees)} 棵树；可更改抽样和问题再建下一棵。</p>'
        add.disabled = len(state.trees) == 3
        render()

    def on_reset(_):
        state.reset()
        add.disabled = False
        status.value = '<p>已清空；可以重新试一组抽样。</p>'
        render()

    add.on_click(on_add)
    reset.on_click(on_reset)
    render()
    return widgets.VBox([intro, widgets.HBox(draws, layout=widgets.Layout(flex_flow='row wrap')),
                         question, widgets.HBox([add, reset]), status, result])


BOOST_X = (.2, .5, .8)
BOOST_Y = (6., 8., 10.)
BOOST_SPLITS = (.35, .65)


def boost_candidate(prediction, threshold):
    """当前残差上训练指定门槛的回归树桩，返回叶均值和残差 SSE。"""
    prediction = tuple(float(x) for x in prediction)
    if len(prediction) != 3 or not all(isfinite(x) for x in prediction):
        raise ValueError('预测必须是三个有限数值。')
    if threshold not in BOOST_SPLITS:
        raise ValueError('请使用提供的密度门槛。')
    residual = tuple(y - p for y, p in zip(BOOST_Y, prediction))
    left = [i for i, x in enumerate(BOOST_X) if x <= threshold]
    right = [i for i, x in enumerate(BOOST_X) if x > threshold]
    left_mean = sum(residual[i] for i in left) / len(left)
    right_mean = sum(residual[i] for i in right) / len(right)
    adjustment = tuple(left_mean if x <= threshold else right_mean for x in BOOST_X)
    sse = sum((r - a) ** 2 for r, a in zip(residual, adjustment))
    return {'threshold': threshold, 'residual': residual, 'left_mean': left_mean,
            'right_mean': right_mean, 'adjustment': adjustment, 'sse': sse}


class BoostingState:
    """记录每一轮的预测；每次都从当前预测重新计算残差。"""
    def __init__(self):
        self.reset()

    def reset(self):
        base = sum(BOOST_Y) / len(BOOST_Y)
        self.prediction = (base,) * len(BOOST_Y)
        self.rounds = []

    def candidate(self, threshold):
        return boost_candidate(self.prediction, threshold)

    def add_round(self, threshold, learning_rate):
        rate = float(learning_rate)
        if not isfinite(rate) or not 0 < rate <= 1:
            raise ValueError('学习率须大于 0 且不超过 1。')
        if len(self.rounds) >= 6:
            raise ValueError('已完成六轮，请重置后再试。')
        candidate = self.candidate(threshold)
        before = self.prediction
        after = tuple(p + rate * a for p, a in zip(before, candidate['adjustment']))
        self.prediction = after
        record = {**candidate, 'before': before, 'rate': rate, 'after': after}
        self.rounds.append(record)
        return record


def _nums(values):
    return '，'.join(f'{v:.3f}'.rstrip('0').rstrip('.') for v in values)


def boosting_lab():
    """构造逐轮提升互动控件；返回 VBox，不依赖其他实验。"""
    state = BoostingState()
    intro = widgets.HTML(value=(
        '<h3>动手让小树纠正甜度预测</h3>'
        '<p>三个模拟瓜的密度依次为 0.2、0.5、0.8，真实甜度为 6、8、10。'
        '先全部预测平均值 8。每轮先算残差＝真实值−当前预测，'
        '再从两个问题中任选一个，让树桩按叶内平均残差给出修正。比较两个候选的误差后，'
        '选择门槛和学习率，点“添加一棵树”。</p>'
        '<p>这演示平方误差回归；分类等其他损失下使用相应的负梯度，不总是普通残差。</p>'))
    current = widgets.HTML(value='')
    threshold = widgets.Dropdown(options=[('密度 ≤ 0.35', .35), ('密度 ≤ 0.65', .65)],
                                 value=.35, description='问题',
                                 layout=widgets.Layout(width='250px'), style={'description_width': '45px'})
    rate = widgets.FloatSlider(value=.5, min=.1, max=1., step=.1, description='学习率',
                               continuous_update=False, readout_format='.1f',
                               style={'description_width': '55px'})
    preview = widgets.HTML(value='')
    add = widgets.Button(description='添加一棵树', button_style='primary')
    reset = widgets.Button(description='重新开始')
    status = widgets.HTML(value='')
    history = widgets.HTML(value='')

    def render(*_):
        residual = tuple(y - p for y, p in zip(BOOST_Y, state.prediction))
        current.value = ('<p><strong>当前预测：</strong>' + _nums(state.prediction)
                         + '；<strong>当前残差：</strong>' + _nums(residual) + '。</p>')
        candidates = [state.candidate(t) for t in BOOST_SPLITS]
        rows = []
        for c in candidates:
            rows.append((f"密度 ≤ {c['threshold']:.2f}", f"{c['left_mean']:.3f}",
                         f"{c['right_mean']:.3f}", _nums(c['adjustment']), f"{c['sse']:.3f}"))
        chosen = state.candidate(threshold.value)
        next_prediction = tuple(p + rate.value * a for p, a in zip(state.prediction, chosen['adjustment']))
        preview.value = (_table(['候选问题', '左叶修正', '右叶修正', '三个瓜各自修正', '拟合残差的平方误差和'], rows)
                         + f'<p>当前选择加上学习率 {rate.value:.1f} 后，预测会变为 <b>{_nums(next_prediction)}</b>。'
                           '“拟合残差的平方误差和”越小，说明这个树桩在当前训练样本上拟合得越好。</p>')
        if state.rounds:
            history.value = _table(['轮次', '问题', '学习率', '该树修正', '更新后预测'],
                                   [(i, f"密度 ≤ {r['threshold']:.2f}", f"{r['rate']:.1f}",
                                     _nums(r['adjustment']), _nums(r['after']))
                                    for i, r in enumerate(state.rounds, 1)])
        else:
            history.value = '<p>还没有添加树；选择问题后先预测会发生什么。</p>'

    def on_add(_):
        try:
            state.add_round(threshold.value, rate.value)
        except ValueError as exc:
            status.value = '<p>' + escape(str(exc)) + '</p>'
            return
        status.value = f'<p>已完成第 {len(state.rounds)} 轮；表格已按新残差重新计算。</p>'
        add.disabled = len(state.rounds) == 6
        render()

    def on_reset(_):
        state.reset()
        add.disabled = False
        status.value = '<p>已回到三瓜的平均甜度。</p>'
        render()

    threshold.observe(render, names='value')
    rate.observe(render, names='value')
    add.on_click(on_add)
    reset.on_click(on_reset)
    render()
    return widgets.VBox([intro, current, preview, threshold, rate, widgets.HBox([add, reset]), status, history])
