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
    return '<div style="width:100%;overflow-x:auto"><table style="border-collapse:collapse;margin:6px 0;min-width:680px"><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def _sample_melon(x, y, name):
    """Draw one occurrence, including duplicates, from the actual sample."""
    melon = MELONS[name]
    peel = '#286948' if melon['color'] == '青绿' else '#23443e'
    stripe = '#83b66e' if melon['color'] == '青绿' else '#5a8a6a'
    label = '好' if melon['good'] else '坏'
    badge = '#16694e' if melon['good'] else '#a43e31'
    return (
        f'<g><circle cx="{x}" cy="{y}" r="16" fill="{peel}" stroke="#173f35" stroke-width="2"/>'
        f'<path d="M {x-6} {y-14} Q {x-11} {y} {x-6} {y+14} M {x+6} {y-14} Q {x+11} {y} {x+6} {y+14}" '
        f'fill="none" stroke="{stripe}" stroke-width="3"/>'
        f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="14" font-weight="bold" fill="white">{escape(name)}</text>'
        f'<text x="{x}" y="{y+31}" text-anchor="middle" font-size="13" font-weight="bold" fill="{badge}">{label}</text></g>'
    )


def _forest_visual(trees, pending_sample, pending_question):
    """Illustrate selected draws, each fitted stump and the real aggregation."""
    parts = [
        '<p style="font-size:13px;color:#526b60">屏幕较窄时，可横向滚动查看完整图形。</p><div style="width:100%;overflow-x:auto"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 455" '
        'role="img" aria-label="森林图：当前抽样预览、每棵树实际抽到的瓜和好瓜概率，以及森林平均概率" '
        'style="display:block;width:100%;min-width:760px;max-width:1000px;height:auto;font-family:system-ui,sans-serif">',
        '<rect width="1000" height="455" rx="14" fill="#f7fbf7"/>',
        '<text x="20" y="29" font-size="20" font-weight="bold" fill="#193f35">把抽到的瓜送进小树</text>',
        '<text x="20" y="51" font-size="14" fill="#445f57">瓜皮颜色＝色泽；瓜下“好／坏”＝已知标签。重复出现代表同一颗瓜抽中多次。</text>',
        '<rect x="15" y="65" width="970" height="75" rx="10" fill="#fff6df" stroke="#d6b25b"/>',
        '<text x="30" y="90" font-size="15" font-weight="bold" fill="#604b22">当前抽样预览</text>',
    ]
    for j, name in enumerate(pending_sample):
        parts.append(_sample_melon(190 + j * 55, 95, name))
    parts.append(f'<text x="440" y="96" font-size="15" fill="#193f35">问题：{escape(pending_question)}</text>')
    note = '已种满三棵树；重新开始后可以再试。' if len(trees) == 3 else '点“种一棵树”后，这组抽样才会进入森林。'
    parts.append(f'<text x="440" y="119" font-size="13" fill="#5b6d65">{note}</text>')
    for i in range(3):
        top = 151 + i * 79
        parts.append(f'<rect x="15" y="{top}" width="970" height="69" rx="10" fill="white" stroke="#cfdfd6"/>')
        parts.append(f'<text x="30" y="{top+27}" font-size="16" font-weight="bold" fill="#193f35">树 {i+1}</text>')
        if i >= len(trees):
            parts.append(f'<text x="150" y="{top+39}" font-size="15" fill="#71857b">等待抽样和种树</text>')
            continue
        tree = trees[i]
        for j, name in enumerate(tree['sample']):
            parts.append(_sample_melon(120+j*51, top+28, name))
        parts.append(f'<text x="355" y="{top+23}" font-size="14" fill="#193f35">{escape(tree["question"])}</text>')
        if tree['did_split']:
            # Highlight the branch taken by the same fixed new melon.
            lx, rx, rootx = 515, 555, 535
            selected = lx if tree['new_goes_yes'] else rx
            parts.append(f'<path d="M {rootx} {top+30} L {lx} {top+49} M {rootx} {top+30} L {rx} {top+49}" stroke="#b7c7be" stroke-width="3" fill="none"/>')
            parts.append(f'<path d="M {rootx} {top+30} L {selected} {top+49}" stroke="#de725d" stroke-width="4" fill="none"/>')
            parts.append(f'<circle cx="{rootx}" cy="{top+30}" r="6" fill="#1c806a"/><circle cx="{lx}" cy="{top+49}" r="6" fill="{ "#de725d" if selected == lx else "#bbc9c0" }"/><circle cx="{rx}" cy="{top+49}" r="6" fill="{ "#de725d" if selected == rx else "#bbc9c0" }"/>')
            parts.append(f'<text x="575" y="{top+44}" font-size="13" fill="#674e44">新瓜走“{ "是" if tree["new_goes_yes"] else "否" }”</text>')
        else:
            parts.append(f'<circle cx="535" cy="{top+35}" r="8" fill="#de725d"/><text x="555" y="{top+40}" font-size="13" fill="#674e44">无法分组，保留根叶</text>')
        p = tree['probability']
        parts.append(f'<text x="750" y="{top+24}" font-size="14" fill="#193f35">好瓜概率 {p:.3f}</text>')
        parts.append(f'<rect x="750" y="{top+34}" width="190" height="14" rx="7" fill="#e5eee7"/><rect x="750" y="{top+34}" width="{190*p:.2f}" height="14" rx="7" fill="#20856b"/>')
    if trees:
        avg = sum(t['probability'] for t in trees) / len(trees)
        formula = ' + '.join(f'{t["probability"]:.3f}' for t in trees)
        parts.append(f'<text x="26" y="422" font-size="16" font-weight="bold" fill="#193f35">合并：({formula}) ÷ {len(trees)} = {avg:.3f}</text>')
    else:
        parts.append('<text x="26" y="422" font-size="16" fill="#526b60">种下第一棵树后，这里会显示森林的平均好瓜概率。</text>')
    parts.append('</svg></div>')
    return ''.join(parts)


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
        picture = _forest_visual(state.trees, [d.value for d in draws], question.value)
        if not state.trees:
            result.value = picture + '<p>还没有树：先选择四次抽样与一个问题。</p>'
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
        result.value = (picture + _table(['树', '抽样', '问题', '新瓜到达', '概率怎么算', '好瓜概率', '单树判断'], rows)
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
        status.value = (f'<p>已建 {len(state.trees)} 棵树；可更改抽样和问题再建下一棵。</p>'
                        if len(state.trees) < 3 else '<p>已建 3 棵树；可以比较结果，或重新开始。</p>')
        add.disabled = len(state.trees) == 3
        render()

    def on_reset(_):
        state.reset()
        add.disabled = False
        status.value = '<p>已清空；可以重新试一组抽样。</p>'
        render()

    add.on_click(on_add)
    reset.on_click(on_reset)
    for control in (*draws, question):
        control.observe(lambda change: render(), names='value')
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


def _boost_visual(prediction, rounds):
    """Draw true sweetness, current prediction and signed residual for each melon."""
    low = min(0., *prediction, *BOOST_Y)
    high = max(12., *prediction, *BOOST_Y)
    start, width = 270., 430.
    scale = width / (high - low)
    origin = start + (0. - low) * scale

    def position(value):
        return start + (value - low) * scale

    def bar(value, y, color):
        end = position(value)
        x, w = min(origin, end), abs(end - origin)
        return f'<rect x="{x:.2f}" y="{y}" width="{w:.2f}" height="14" rx="5" fill="{color}"/>'

    pieces = [
        '<p style="font-size:13px;color:#526b60">屏幕较窄时，可横向滚动查看完整图形。</p><div style="width:100%;overflow-x:auto"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 335" '
        'role="img" aria-label="三颗模拟瓜的真实甜度、当前预测和真实减预测的残差比较图" '
        'style="display:block;width:100%;min-width:760px;max-width:1000px;height:auto;font-family:system-ui,sans-serif">',
        '<rect width="1000" height="335" rx="14" fill="#f7fbf7"/>',
        f'<text x="22" y="30" font-size="20" font-weight="bold" fill="#193f35">第 {rounds} 轮：看看每颗瓜离真实甜度还有多远</text>',
        '<rect x="22" y="45" width="15" height="12" rx="3" fill="#22856c"/><text x="45" y="56" font-size="14" fill="#193f35">真实甜度</text>',
        '<rect x="140" y="45" width="15" height="12" rx="3" fill="#d96f50"/><text x="164" y="56" font-size="14" fill="#193f35">当前预测</text>',
        '<text x="285" y="56" font-size="14" fill="#526b60">残差 = 真实 − 预测；条形越接近，差距越小。</text>',
    ]
    for i, (density, actual, pred) in enumerate(zip(BOOST_X, BOOST_Y, prediction)):
        top = 72 + i * 80
        residual = actual - pred
        direction = '预测偏低' if residual > 1e-9 else '预测偏高' if residual < -1e-9 else '预测相同'
        signed = f'{residual:+.3f}'.rstrip('0').rstrip('.') if abs(residual) > 1e-9 else '0'
        pieces.append(f'<rect x="15" y="{top}" width="970" height="69" rx="10" fill="white" stroke="#cfdfd6"/>')
        # These three melons represent the regression samples, not the good/bad labels in the forest.
        pieces.append(f'<circle cx="54" cy="{top+32}" r="20" fill="#286948" stroke="#173f35" stroke-width="2"/><path d="M 46 {top+15} Q 42 {top+32} 46 {top+49} M 62 {top+15} Q 66 {top+32} 62 {top+49}" fill="none" stroke="#83b66e" stroke-width="3"/>')
        pieces.append(f'<text x="54" y="{top+37}" text-anchor="middle" font-size="14" font-weight="bold" fill="white">{chr(65+i)}</text>')
        pieces.append(f'<text x="92" y="{top+29}" font-size="15" font-weight="bold" fill="#193f35">瓜 {chr(65+i)}</text><text x="92" y="{top+50}" font-size="13" fill="#526b60">密度 {density:g}</text>')
        pieces.append(f'<text x="182" y="{top+25}" font-size="13" fill="#193f35">真实</text><text x="182" y="{top+49}" font-size="13" fill="#7d3b2b">预测</text>')
        pieces.append(f'<rect x="{start}" y="{top+13}" width="{width}" height="14" rx="5" fill="#e9f0ea"/><rect x="{start}" y="{top+37}" width="{width}" height="14" rx="5" fill="#f5ede8"/>')
        pieces.append(bar(actual, top+13, '#22856c'))
        pieces.append(bar(pred, top+37, '#d96f50'))
        pieces.append(f'<text x="715" y="{top+25}" font-size="14" fill="#193f35">{actual:g}</text><text x="715" y="{top+49}" font-size="14" fill="#7d3b2b">{_nums((pred,))}</text>')
        pieces.append(f'<text x="790" y="{top+28}" font-size="15" font-weight="bold" fill="#193f35">残差 {signed}</text><text x="790" y="{top+49}" font-size="13" fill="#526b60">{direction}</text>')
    pieces.append(f'<text x="270" y="326" font-size="13" fill="#526b60">条形刻度：{low:g} 至 {high:g}（虚构的教学甜度）</text></svg></div>')
    return ''.join(pieces)


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
        current.value = (_boost_visual(state.prediction, len(state.rounds))
                         + '<p><strong>当前预测：</strong>' + _nums(state.prediction)
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
        status.value = (f'<p>已完成第 {len(state.rounds)} 轮；表格已按新残差重新计算。</p>'
                        + ('<p>本实验最多六轮；如需继续探索，请重新开始。</p>' if len(state.rounds) == 6 else ''))
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
