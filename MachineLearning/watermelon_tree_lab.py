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


def watermelon_tree_lab():
    """返回可显示的互动面板：选择节点和问题，分裂、定叶或重置。"""
    tree = ManualTree()
    title = widgets.HTML('<h3>亲手种一棵西瓜决策树</h3>'
                         '<p>先选一个待处理节点，再比较问题。选一个问题分裂；'
                         '也可以停止并把当前组定为叶子。完成后会按树图先“是”后“否”的顺序自动选择下一个待处理节点；'
                         '也可以用“当前节点”菜单自己切换。数据是教学模拟，不是挑瓜规则。</p>')
    rows = ''.join('<tr>' + ''.join(f'<td style="padding:2px 8px">{escape(str(x))}</td>' for x in (m[0], m[1], m[2], m[3], '好' if m[4] else '坏')) + '</tr>' for m in MELONS)
    table = widgets.HTML('<p><b>训练瓜</b>（瓜蒂为本实验补充的特征）</p>'
                         '<table><thead><tr><th>编号</th><th>色泽</th><th>纹理</th><th>瓜蒂</th><th>已知好坏</th></tr></thead><tbody>' + rows + '</tbody></table>')
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

    def update_prediction():
        sample = {'color': new_color.value, 'texture': new_texture.value, 'stem': new_stem.value}
        trace, good, total, done = tree.predict(sample)
        path = ' → '.join(f'{FEATURES[f][0]}是{FEATURES[f][1]}：{"是" if a else "否"}' for f, a in trace)
        status = '叶子已确定' if done else '当前节点尚未定为叶子，预测只是暂时的'
        result = ('好坏平票，无法仅凭多数票确定' if good * 2 == total else
                  '预测好瓜' if good * 2 > total else '预测坏瓜')
        new_output.value = (f'<b>新瓜走的路：</b>{escape(path or "尚未提问")}'
                            f'<br>{status}；该组好瓜 {good}/{total}，{result}。')

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
        preview.value = (f'<p><b>当前节点：</b>{escape(_path_name(tree, path))}。'
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
        tree_view.value = '<h4>你建出的树</h4><ul>' + _tree_html(tree.root, path) + '</ul>'
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
        tree_view.value = '<h4>你建出的树</h4><ul>' + _tree_html(tree.root, node_select.value) + '</ul>'
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
    return widgets.VBox([title, table, node_select, question, preview,
                         widgets.HBox([split, finish, reset]), message, tree_view,
                         widgets.HTML('<h4>让一颗新瓜走走看</h4><p>改变特征，观察它经过的问题和到达的叶子。</p>'),
                         widgets.HBox([new_color, new_texture, new_stem]), new_output])
