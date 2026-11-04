"""西瓜课的手动建树实验，使用同一份小数据计算每一步。

在课程笔记本中运行：
    from watermelon_tree_lab import watermelon_tree_lab
    display(watermelon_tree_lab())
"""
from dataclasses import dataclass, field
from html import escape

import ipywidgets as widgets


# 色泽、纹理及好坏与课程中的六瓜算例相同；瓜蒂是本实验增加的虚构特征。
MELONS = (
    ('A', '青绿', '清晰', '蜷缩', 1),
    ('B', '青绿', '清晰', '蜷缩', 1),
    ('C', '青绿', '清晰', '硬挺', 0),
    ('D', '乌黑', '清晰', '硬挺', 1),
    ('E', '乌黑', '模糊', '硬挺', 0),
    ('F', '乌黑', '模糊', '蜷缩', 0),
)
FEATURES = {'color': ('色泽', '青绿', '乌黑'),
            'texture': ('纹理', '清晰', '模糊'),
            'stem': ('瓜蒂', '蜷缩', '硬挺')}
INDEX = {'color': 1, 'texture': 2, 'stem': 3}


def gini(indices):
    """返回非空节点的不纯度；空节点没有定义。"""
    if not indices:
        raise ValueError('空节点无法计算基尼不纯度')
    n = len(indices)
    good = sum(MELONS[i][4] for i in indices)
    return 1.0 - (good / n) ** 2 - ((n - good) / n) ** 2


@dataclass
class Node:
    indices: tuple
    feature: str = None
    children: dict = field(default_factory=dict)
    closed: bool = False


class ManualTree:
    """存放学习者做出的划分；所有统计直接从节点中样本计算。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.root = Node(tuple(range(len(MELONS))))

    def node(self, path):
        node = self.root
        for answer in path:
            if node.feature is None or answer not in node.children:
                raise ValueError('不存在这个节点')
            node = node.children[answer]
        return node

    def pending(self):
        result = []
        def visit(node, path):
            if node.feature is None:
                if not node.closed:
                    result.append(path)
            else:
                visit(node.children[True], path + (True,))
                visit(node.children[False], path + (False,))
        visit(self.root, ())
        return result

    def partition(self, path, feature):
        node = self.node(path)
        if feature not in FEATURES:
            raise ValueError('未知的候选问题')
        yes_value = FEATURES[feature][1]
        yes = tuple(i for i in node.indices if MELONS[i][INDEX[feature]] == yes_value)
        no = tuple(i for i in node.indices if MELONS[i][INDEX[feature]] != yes_value)
        return yes, no

    def score(self, path, feature):
        yes, no = self.partition(path, feature)
        if not yes or not no:
            return None
        n = len(yes) + len(no)
        return (len(yes) * gini(yes) + len(no) * gini(no)) / n

    def split(self, path, feature):
        node = self.node(path)
        if node.feature is not None or node.closed:
            raise ValueError('请先选择一个尚未完成的叶节点')
        yes, no = self.partition(path, feature)
        if not yes or not no:
            raise ValueError('这个问题不能把当前的瓜分成非空的两组')
        node.feature = feature
        node.children = {True: Node(yes), False: Node(no)}

    def finish(self, path):
        node = self.node(path)
        if node.feature is not None or node.closed:
            raise ValueError('请先选择一个尚未完成的叶节点')
        node.closed = True

    def predict(self, sample):
        """返回新瓜路径及叶子统计；未关闭节点标为尚未完成。"""
        node = self.root
        trace = []
        while node.feature is not None:
            feature = node.feature
            answer = sample[feature] == FEATURES[feature][1]
            trace.append((feature, answer))
            node = node.children[answer]
        good = sum(MELONS[i][4] for i in node.indices)
        return trace, good, len(node.indices), node.closed


def _path_name(tree, path):
    """Describe every question on the way, rather than ambiguous yes/no labels."""
    node = tree.root
    steps = ['根节点']
    for answer in path:
        name, value, _ = FEATURES[node.feature]
        steps.append(f'{name}是{value}：{"是" if answer else "否"}')
        node = node.children[answer]
    return ' → '.join(steps)


def _counts(indices):
    good = sum(MELONS[i][4] for i in indices)
    return f'{len(indices)} 个（好 {good}，坏 {len(indices) - good}）'


def _ids(indices):
    return '、'.join(MELONS[i][0] for i in indices) or '无'


def _leaf_text(indices):
    good = sum(MELONS[i][4] for i in indices)
    bad = len(indices) - good
    if good == bad:
        result = '好坏平票，不能仅凭多数票确定'
    else:
        result = '预测好瓜' if good > bad else '预测坏瓜'
    return f'{result}（好 {good} / {len(indices)}）'


def _tree_html(node, selected=None, path=()):
    counts = escape(_counts(node.indices))
    ids = escape(_ids(node.indices))
    if node.feature is None:
        status = '已定叶子' if node.closed else '待处理节点'
        current = ' <b>← 当前节点</b>' if selected == path and not node.closed else ''
        return f'<li><b>{status}</b>：{ids}，{counts}；{escape(_leaf_text(node.indices))}{current}</li>'
    name, value, _ = FEATURES[node.feature]
    yes = _tree_html(node.children[True], selected, path + (True,))
    no = _tree_html(node.children[False], selected, path + (False,))
    return (f'<li><b>{escape(name)}是{escape(value)}吗？</b>（{ids}，{counts}）'
            f'<ul><li>是<ul>{yes}</ul></li><li>否<ul>{no}</ul></li></ul></li>')


def _melon(x, y, index, size=34):
    """Draw one labeled sample. Flesh color is a label, not its skin color."""
    name, _, _, _, good = MELONS[index]
    flesh = '#eb8078' if good else '#9ca9ae'
    mark = '好' if good else '坏'
    return (f'<g transform="translate({x},{y})">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="#2d8065"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{size*.43}" fill="#e9f4df"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{size*.39}" fill="{flesh}"/>'
            f'<circle cx="{size*.30}" cy="{size*.32}" r="1.1" fill="#495e54"/>'
            f'<circle cx="{size*.70}" cy="{size*.32}" r="1.1" fill="#495e54"/>'
            f'<circle cx="{size*.30}" cy="{size*.70}" r="1.1" fill="#495e54"/>'
            f'<circle cx="{size*.70}" cy="{size*.70}" r="1.1" fill="#495e54"/>'
            f'<text x="{size/2}" y="{size*.66}" text-anchor="middle" '
            f'font-size="{size*.45}" font-weight="bold" fill="#fff">{escape(name)}</text>'
            f'<text x="{size/2}" y="{size+15}" text-anchor="middle" '
            f'font-size="13" fill="#243c42">{mark}</text></g>')


def _wrap_svg(svg, min_width=600):
    """Keep dense diagrams legible on a narrow notebook; allow horizontal scroll."""
    return (f'<div style="width:100%;max-width:100%;overflow-x:auto;">'
            f'<div style="min-width:{min_width}px;">{svg}</div></div>'
            '<small style="color:#49655d">图较宽时，可在图内左右滑动查看。</small>')


def _split_svg(node, yes, no, score, question):
    """Preview only; this figure must not change the saved tree."""
    width, height = 760, 335
    font = 'font-family="system-ui, sans-serif"'
    label, wanted, _ = FEATURES[question]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
             f'data-role="split-preview" data-yes="{escape(_ids(yes))}" data-no="{escape(_ids(no))}" '
             f'role="img" aria-label="候选问题的分组预览，尚未加入决策树" '
             f'style="display:block;width:100%;height:auto" {font}>',
             '<rect x="2" y="2" width="756" height="331" rx="16" fill="#f7faf8" stroke="#d1e3dc"/>',
             '<text x="380" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#20433b">① 当前节点的训练瓜</text>']
    start = (width - len(node.indices) * 55) / 2
    for j, index in enumerate(node.indices):
        parts.append(_melon(start + j * 55, 42, index, 32))
    parts += [f'<text x="380" y="120" text-anchor="middle" font-size="15" fill="#294b45">'
              f'预览问题：{escape(label)}是{escape(wanted)}吗？</text>',
              '<path d="M380 130 L195 168 M380 130 L565 168" fill="none" stroke="#829c93" stroke-width="2"/>']
    for x, values, answer, color in ((20, yes, '是', '#2a846e'), (395, no, '否', '#bf665b')):
        parts.append(f'<rect x="{x}" y="168" width="345" height="136" rx="12" fill="#fff" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{x+16}" y="192" font-size="16" font-weight="bold" fill="{color}">回答「{answer}」</text>')
        if values:
            first = x + (345 - len(values) * 49) / 2
            for j, index in enumerate(values):
                parts.append(_melon(first + j * 49, 202, index, 30))
            parts.append(f'<text x="{x+172}" y="287" text-anchor="middle" font-size="14" fill="#28433d">'
                         f'{escape(_counts(values))} · 基尼 {gini(values):.3f}</text>')
        else:
            parts.append(f'<text x="{x+172}" y="250" text-anchor="middle" font-size="15" fill="#7a5147">这一边没有瓜</text>')
    footer = '两边都需要有瓜，才能用这个问题分裂' if score is None else f'两组加权基尼：{score:.3f}'
    parts += [f'<text x="380" y="325" text-anchor="middle" font-size="14" fill="#32534a">{footer}</text>', '</svg>']
    return _wrap_svg(''.join(parts), min_width=600)


def _tree_svg(tree, selected, sample_path):
    """Draw the committed tree, using actual nodes and the current prediction path."""
    positions, nodes, leaves = {}, {}, []

    def collect(node, path=(), depth=0):
        nodes[path] = (node, depth)
        if node.feature is None:
            leaves.append(path)
        else:
            collect(node.children[True], path + (True,), depth + 1)
            collect(node.children[False], path + (False,), depth + 1)

    collect(tree.root)
    width = max(640, len(leaves) * 185)
    height = 90 + max(d for _, d in nodes.values()) * 158 + 115
    for i, path in enumerate(leaves):
        positions[path] = width * (i + .5) / len(leaves)

    def center(node, path=()):
        if node.feature is None:
            return positions[path]
        left = center(node.children[True], path + (True,))
        right = center(node.children[False], path + (False,))
        positions[path] = (left + right) / 2
        return positions[path]

    center(tree.root)
    route = {sample_path[:i] for i in range(len(sample_path) + 1)}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
             f'data-role="committed-tree" '
             f'role="img" aria-label="已建立的决策树，描边标出当前节点和新瓜路径" '
             f'style="display:block;width:100%;height:auto" font-family="system-ui, sans-serif">',
             f'<rect x="2" y="2" width="{width-4}" height="{height-4}" rx="16" fill="#f7faf8" stroke="#d1e3dc"/>']
    for path, (node, depth) in nodes.items():
        if node.feature is None:
            continue
        x, y = positions[path], 42 + depth * 158
        for answer in (True, False):
            child = path + (answer,)
            tx, ty = positions[child], y + 158
            active = child in route
            stroke = '#d39528' if active else '#9cafaa'
            line_width = 5 if active else 2
            parts.append(f'<path d="M{x:.1f} {y+102} L{tx:.1f} {ty}" fill="none" stroke="{stroke}" stroke-width="{line_width}"/>')
            parts.append(f'<text x="{(x+tx)/2:.1f}" y="{y+132}" text-anchor="middle" '
                         f'font-size="14" font-weight="bold" fill="#35554c">{"是" if answer else "否"}</text>')
    for path, (node, depth) in nodes.items():
        x, y = positions[path] - 79, 42 + depth * 158
        active, current = path in route, path == selected and not node.closed
        path_code = 'root' + ''.join('/yes' if answer else '/no' for answer in path)
        parts.append(f'<g data-node="{path_code}" data-selected="{str(current).lower()}" '
                     f'data-route="{str(active).lower()}" data-samples="{escape(_ids(node.indices))}">')
        outline = '#d39528' if active else '#b5cbc2'
        fill = '#e7f5f0' if current else '#fff'
        parts.append(f'<rect x="{x:.1f}" y="{y}" width="158" height="102" rx="12" '
                     f'fill="{fill}" stroke="{outline}" stroke-width="{4 if active else 2}"/>')
        if current:
            parts.append(f'<rect x="{x+5:.1f}" y="{y+5}" width="148" height="92" rx="9" fill="none" stroke="#197c83" stroke-width="2" stroke-dasharray="4 3"/>')
        if node.feature is not None:
            name, wanted, _ = FEATURES[node.feature]
            label = f'{name}是{wanted}？'
            status = '问题节点'
        else:
            label = '已定叶子' if node.closed else '待处理 · 暂定'
            good = sum(MELONS[i][4] for i in node.indices)
            status = '好坏平票' if good*2 == len(node.indices) else ('预测好瓜' if good*2 > len(node.indices) else '预测坏瓜')
        parts.append(f'<text x="{x+79:.1f}" y="{y+24}" text-anchor="middle" font-size="15" font-weight="bold" fill="#22453d">{escape(label)}</text>')
        parts.append(f'<text x="{x+79:.1f}" y="{y+46}" text-anchor="middle" font-size="13" fill="#4d6962">{escape(status)}</text>')
        start = x + (158 - len(node.indices) * 23) / 2
        for j, idx in enumerate(node.indices):
            mark = '好' if MELONS[idx][4] else '坏'
            fill_chip = '#f4beb7' if MELONS[idx][4] else '#c6d0d3'
            cx = start + j * 23 + 10
            shape = (f'<circle cx="{cx:.1f}" cy="{y+69}" r="9" fill="{fill_chip}" stroke="#56816d"/>'
                     if MELONS[idx][4] else
                     f'<rect x="{cx-9:.1f}" y="{y+60}" width="18" height="18" rx="2" fill="{fill_chip}" stroke="#56816d"/>')
            parts.append(shape + f'<text x="{cx:.1f}" y="{y+73}" text-anchor="middle" font-size="10" fill="#243c42">{escape(MELONS[idx][0])}</text>'
                         f'<title>{escape(MELONS[idx][0])}：{mark}瓜</title>')
        if current:
            parts.append(f'<text x="{x+79:.1f}" y="{y+95}" text-anchor="middle" font-size="12" font-weight="bold" fill="#126b73">当前节点</text>')
        parts.append('</g>')
    parts.append('</svg>')
    return _wrap_svg(''.join(parts), min_width=min(width, 680))


def _prediction_svg(tree, sample):
    """Small horizontal route repeated beside the new-melon controls."""
    trace, good, total, closed = tree.predict(sample)
    width = max(620, 170 * (len(trace) + 2))
    result = '好坏平票' if good * 2 == total else ('预测好瓜' if good * 2 > total else '预测坏瓜')
    status = '已定叶子' if closed else '暂定结果'
    box_width = width / (len(trace) + 2)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 118" '
             f'data-role="prediction-route" '
             f'role="img" aria-label="新瓜沿已建立的树经过的问题与当前结果" '
             f'style="display:block;width:100%;height:auto" font-family="system-ui, sans-serif">']
    labels = [('新瓜', f'{sample["color"]} · {sample["texture"]} · {sample["stem"]}')]
    labels += [(f'{FEATURES[feature][0]}是{FEATURES[feature][1]}？', '是' if answer else '否')
               for feature, answer in trace]
    labels.append((result, f'{status} · 好瓜 {good}/{total}'))
    for i, (top, bottom) in enumerate(labels):
        x = i * box_width + 5
        if i:
            parts.append(f'<path d="M{x-18:.1f} 57 L{x-3:.1f} 57 '
                         f'M{x-9:.1f} 52 L{x-3:.1f} 57 L{x-9:.1f} 62" '
                         'fill="none" stroke="#c58c25" stroke-width="3"/>')
        fill = '#fff3cd' if i == len(labels)-1 else '#fff'
        parts.append(f'<rect x="{x:.1f}" y="18" width="{box_width-24:.1f}" height="76" rx="12" fill="{fill}" stroke="#d6a742" stroke-width="2"/>')
        parts.append(f'<text x="{x+(box_width-24)/2:.1f}" y="47" text-anchor="middle" font-size="14" fill="#26443c">{escape(top)}</text>')
        parts.append(f'<text x="{x+(box_width-24)/2:.1f}" y="73" text-anchor="middle" font-size="13" font-weight="bold" fill="#76521a">{escape(bottom)}</text>')
    parts.append('</svg>')
    return _wrap_svg(''.join(parts), min_width=min(width, 620))


def watermelon_tree_lab():
    """返回可显示的互动面板：选择节点和问题，分裂、定叶或重置。"""
    tree = ManualTree()
    title = widgets.HTML('<style>.watermelon-tree-lab .widget-html-content '
                         '{min-width:0;max-width:100%;width:100%;}</style>'
                         '<h3>亲手种一棵西瓜决策树</h3>'
                         '<p>先选一个待处理节点，再比较问题。选一个问题分裂；'
                         '也可以停止并把当前组定为叶子。完成后会按树图先“是”后“否”的顺序自动选择下一个待处理节点；'
                         '也可以用“当前节点”菜单自己切换。数据是教学模拟，不是挑瓜规则。</p>')
    rows = ''.join('<tr>' + ''.join(f'<td style="padding:2px 8px">{escape(str(x))}</td>' for x in (m[0], m[1], m[2], m[3], '好' if m[4] else '坏')) + '</tr>' for m in MELONS)
    table = widgets.HTML('<p><b>训练瓜 A–F</b>：瓜蒂是本实验补充的模拟特征。'
                         '图中瓜瓤颜色和“好/坏”表示已知标签，<b>不是外皮色泽</b>。</p>'
                         '<details><summary>查看六个瓜的完整特征表</summary>'
                         '<table><thead><tr><th>编号</th><th>色泽</th><th>纹理</th><th>瓜蒂</th><th>已知好坏</th></tr></thead><tbody>' + rows + '</tbody></table></details>')
    node_select = widgets.Dropdown(description='当前节点', layout=widgets.Layout(width='95%'))
    question = widgets.Dropdown(options=[(f'{v[0]}是{v[1]}吗？', k) for k, v in FEATURES.items()],
                                value='texture', description='候选问题', layout=widgets.Layout(width='95%'))
    preview = widgets.HTML()
    split = widgets.Button(description='用这个问题分裂', button_style='success')
    finish = widgets.Button(description='停止，定为叶子')
    reset = widgets.Button(description='重新开始')
    message = widgets.HTML()
    tree_view = widgets.HTML()
    new_color = widgets.Dropdown(options=['青绿', '乌黑'], description='色泽')
    new_texture = widgets.Dropdown(options=['清晰', '模糊'], description='纹理')
    new_stem = widgets.Dropdown(options=['蜷缩', '硬挺'], description='瓜蒂')
    new_output = widgets.HTML()
    node_buttons = widgets.VBox()

    def refresh_node_buttons():
        """Offer direct, keyboard-accessible actions next to the drawn tree."""
        choices = []
        for path in tree.pending():
            is_current = path == node_select.value
            ids = _ids(tree.node(path).indices)
            button = widgets.Button(
                description=f'{"当前" if is_current else "选择"}：{ids}',
                tooltip=_path_name(tree, path),
                button_style='info' if is_current else '',
                layout=widgets.Layout(width='auto', min_width='120px'))
            button.on_click(lambda _, p=path: setattr(node_select, 'value', p))
            choices.append(button)
        hint = ('<b>选择树中要继续处理的节点：</b>按钮中的字母是该节点的瓜。'
                if choices else '所有节点已处理完成。')
        node_buttons.children = (widgets.HTML(hint),
                                 widgets.HBox(choices, layout=widgets.Layout(flex_flow='row wrap')))

    def update_prediction():
        sample = {'color': new_color.value, 'texture': new_texture.value, 'stem': new_stem.value}
        trace, good, total, done = tree.predict(sample)
        route = tuple(answer for _, answer in trace)
        path = ' → '.join(f'{FEATURES[f][0]}是{FEATURES[f][1]}：{"是" if a else "否"}' for f, a in trace)
        status = '叶子已确定' if done else '当前节点尚未定为叶子，预测只是暂时的'
        result = ('好坏平票，无法仅凭多数票确定' if good * 2 == total else
                  '预测好瓜' if good * 2 > total else '预测坏瓜')
        new_output.value = (_prediction_svg(tree, sample)
                            + f'<b>新瓜走的路：</b>{escape(path or "尚未提问")}'
                            f'<br>{status}；该组好瓜 {good}/{total}，{result}。')
        tree_view.value = ('<h4>② 已建立的树</h4>'
                           '<p>青色虚线：当前节点；金色边框和连线：新瓜的路径。'
                           '粉色圆形为好瓜，灰色方形为坏瓜；图形里的字母对应一颗训练瓜。</p>'
                           + _tree_svg(tree, node_select.value, route)
                           + '<details><summary>查看树的文字说明</summary><ul>'
                           + _tree_html(tree.root, node_select.value) + '</ul></details>')

    def update_preview(*_):
        path = node_select.value
        if path is None:
            preview.value = '<p>没有待处理节点，树已完成。可以在下方观察新瓜的路径，或重新开始。</p>'
            split.disabled = finish.disabled = True
            update_prediction()
            return
        node = tree.node(path)
        yes, no = tree.partition(path, question.value)
        score = tree.score(path, question.value)
        split.disabled = score is None
        finish.disabled = False
        if score is None:
            score_text = '不能分裂：一边没有瓜；请换问题，或把本组定为叶子。'
        else:
            score_text = (f'加权基尼 = {len(yes)}/{len(node.indices)} × {gini(yes):.3f}'
                          f' + {len(no)}/{len(node.indices)} × {gini(no):.3f} = <b>{score:.3f}</b>')
        preview.value = ('<h4>① 候选划分预览（尚未加入树）</h4>'
                         + _split_svg(node, yes, no, score, question.value)
                         + f'<p><b>当前节点：</b>{escape(_path_name(tree, path))}。'
                         f'<br>待处理节点还剩 {len(tree.pending())} 个；当前有瓜 {escape(_ids(node.indices))}，'
                         f'{_counts(node.indices)}，基尼 {gini(node.indices):.3f}。'
                         f'<br>回答「是」：{escape(_ids(yes))}，{_counts(yes)}；'
                         f'回答「否」：{escape(_ids(no))}，{_counts(no)}。'
                         f'<br>{score_text}</p>')
        update_prediction()

    def on_node_selected(change):
        path = change['new']
        if path is not None:
            message.value = (f'<p>已手动切换到【{escape(_path_name(tree, path))}】；'
                             f'还有 {len(tree.pending())} 个待处理节点。</p>')
        refresh_node_buttons()
        update_preview()

    def refresh():
        old = node_select.value
        paths = tree.pending()
        # Updating Dropdown.options can briefly select its first item. Suppress
        # manual-selection feedback until the intended selection is restored.
        node_select.unobserve(on_node_selected, names='value')
        try:
            node_select.options = [
                (f'{_path_name(tree, p)}（瓜 {_ids(tree.node(p).indices)}）', p)
                for p in paths
            ]
            node_select.value = old if old in paths else (paths[0] if paths else None)
            node_select.disabled = not paths
        finally:
            node_select.observe(on_node_selected, names='value')
        refresh_node_buttons()
        update_preview()

    def next_step():
        paths = tree.pending()
        if not paths:
            return '所有节点都已处理，树已完成；可在下方让新瓜走一遍，或重新开始。'
        return (f'还有 {len(paths)} 个待处理节点。为继续处理其他分支，按树图先“是”后“否”的顺序，'
                f'已自动切换到【{escape(_path_name(tree, node_select.value))}】；'
                '也可用“当前节点”菜单选择其他待处理节点。')

    def on_split(_):
        try:
            path = node_select.value
            name = _path_name(tree, path)
            feature, value, _ = FEATURES[question.value]
            tree.split(path, question.value)
            refresh()
            message.value = (f'<p>已将【{escape(name)}】按“{escape(feature)}是{escape(value)}吗？”分成两组。'
                             f'{next_step()}</p>')
        except ValueError as exc:
            message.value = escape(str(exc))

    def on_finish(_):
        try:
            path = node_select.value
            name = _path_name(tree, path)
            ids = _ids(tree.node(path).indices)
            tree.finish(path)
            refresh()
            message.value = (f'<p>已将【{escape(name)}】中的瓜 {escape(ids)} 定为叶子。'
                             f'{next_step()}</p>')
        except ValueError as exc:
            message.value = escape(str(exc))

    def on_reset(_):
        tree.reset()
        refresh()
        message.value = '<p>已重新开始：回到根节点，待处理节点还剩 1 个。</p>'

    node_select.observe(on_node_selected, names='value')
    question.observe(update_preview, names='value')
    split.on_click(on_split)
    finish.on_click(on_finish)
    reset.on_click(on_reset)
    for control in (new_color, new_texture, new_stem):
        control.observe(lambda change: update_prediction(), names='value')
    refresh()
    root = widgets.VBox([title, table, node_select, question, preview,
                         widgets.HBox([split, finish, reset], layout=widgets.Layout(flex_flow='row wrap')),
                         message, tree_view, node_buttons,
                         widgets.HTML('<h4>让一颗新瓜走走看</h4><p>改变特征，观察它经过的问题和到达的叶子。</p>'),
                         widgets.HBox([new_color, new_texture, new_stem], layout=widgets.Layout(flex_flow='row wrap')),
                         new_output])
    root.add_class('watermelon-tree-lab')
    return root
