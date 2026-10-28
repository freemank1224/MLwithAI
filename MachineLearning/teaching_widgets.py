"""课程演示的绘图与 Widget 封装；学习者只需运行笔记本中的演示调用。

算法练习留在笔记本中，并作为参数传入这里，修改后重运行演示即可观察。
"""
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output


def linear_guess(x, y):
    def show_guess(w, b, sample):
        prediction = w * x + b
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(x, y, alpha=.45, color='#4169a1', label='Students')
        ax.plot([0, 8], b + w * np.array([0, 8]), color='#e88b36', lw=2.5, label='Prediction')
        ax.plot([x[sample], x[sample]], [y[sample], prediction[sample]], color='#d33f49', lw=3, label='One residual')
        ax.set(xlim=(0, 8), ylim=(20, 110), xlabel='Study time (hours)', ylabel='Score',
               title=f'Mean squared error = {np.mean((prediction - y)**2):.1f}')
        ax.legend(); ax.grid(alpha=.2); plt.show()

    guess_controls = widgets.interactive(show_guess,
        w=widgets.FloatSlider(value=4, min=0, max=12, step=.2, description='w'),
        b=widgets.FloatSlider(value=30, min=0, max=80, step=1, description='b'),
        sample=widgets.IntSlider(value=0, min=0, max=len(x)-1, description='Student'))
    display(guess_controls)


def linear_training(x, y, trajectories, rates):
    def show_training(step, learning_rate):
        history = trajectories[learning_rate]
        w, b, loss = history[step]
        fig, (ax, loss_ax) = plt.subplots(1, 2, figsize=(11, 4))
        ax.scatter(x, y, alpha=.4, color='#4169a1')
        ax.plot([0, 8], b + w * np.array([0, 8]), color='#e88b36', lw=2.5)
        ax.set(xlim=(0, 8), ylim=(0, 115), xlabel='Study time (hours)', ylabel='Score', title=f'Step {step}: w={w:.2g}, b={b:.2g}')
        loss_ax.semilogy(np.arange(step + 1), history[:step + 1, 2], color='#3c9b85')
        loss_ax.set(xlim=(0, 200), xlabel='Update step', ylabel='MSE (log scale)', title=f'Loss = {loss:.3g}')
        for a in (ax, loss_ax): a.grid(alpha=.2)
        plt.tight_layout(); plt.show()

    play = widgets.Play(value=0, min=0, max=200, step=5, interval=650, description='Play')
    step_slider = widgets.IntSlider(value=0, min=0, max=200, description='Step', continuous_update=False)
    widgets.link((play, 'value'), (step_slider, 'value'))
    rate_selector = widgets.Dropdown(options=rates, value=0.01, description='Learning rate')
    training_output = widgets.interactive_output(show_training, {'step': step_slider, 'learning_rate': rate_selector})
    display(widgets.VBox([widgets.HBox([play, step_slider]), rate_selector, training_output]))


def logistic_learning(X_train, y_train, states, losses, sigmoid):
    def show_learning(step, event):
        w, b = states[step]
        grid = np.linspace(0, 10, 150)
        curve = sigmoid(np.column_stack([grid, np.full_like(grid, event)]) @ w + b)
        mask = X_train[:, 1] == event
        fig, (ax, loss_ax) = plt.subplots(1, 2, figsize=(10, 4))
        ax.scatter(X_train[mask, 0], y_train[mask], alpha=.23, s=20, color='#516b91')
        ax.plot(grid, curve, lw=3, color='#3c9b85')
        ax.set(xlabel='Advertising (hundreds)', ylabel='Probability / outcome', ylim=(-.08, 1.08), title=f'Step {step}')
        loss_ax.plot(np.arange(step + 1), losses[:step + 1], color='#e88b36')
        loss_ax.set(xlim=(0, 500), xlabel='Update step', ylabel='Training log loss', title=f'Loss = {losses[step]:.3f}')
        for a in (ax, loss_ax): a.grid(alpha=.2)
        plt.tight_layout(); plt.show()

    learn_play = widgets.Play(value=0, min=0, max=500, step=20, interval=650, description='Play')
    learn_step = widgets.IntSlider(value=0, min=0, max=500, step=5, description='Step', continuous_update=False)
    widgets.link((learn_play, 'value'), (learn_step, 'value'))
    learn_event = widgets.Dropdown(options=[('No event', 0), ('Weekend event', 1)], description='Event')
    learn_output = widgets.interactive_output(show_learning, {'step': learn_step, 'event': learn_event})
    display(widgets.VBox([widgets.HBox([learn_play, learn_step]), learn_event, learn_output]))


def logistic_threshold(model, X_test, y_test):
    def explore_threshold(threshold, ad_budget, event):
        grid = np.linspace(0, 10, 200)
        probabilities = model.predict_proba(np.column_stack([grid, np.full_like(grid, event)]))[:, 1]
        chosen = model.predict_proba([[ad_budget, event]])[0, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        predicted = (test_prob >= threshold).astype(int)
        tn = np.sum((y_test == 0) & (predicted == 0))
        fp = np.sum((y_test == 0) & (predicted == 1))
        fn = np.sum((y_test == 1) & (predicted == 0))
        tp = np.sum((y_test == 1) & (predicted == 1))
        fig, (ax, matrix_ax) = plt.subplots(1, 2, figsize=(10, 4))
        ax.plot(grid, probabilities, color='#3c9b85', lw=3)
        ax.axhline(threshold, ls='--', color='#d55c55', label='Threshold')
        ax.axvline(ad_budget, ls=':', color='#516b91')
        ax.scatter([ad_budget], [chosen], s=70, color='#e88b36', zorder=3)
        ax.set(xlabel='Advertising (hundreds)', ylabel='Probability of sellout', ylim=(0, 1),
               title=f'Chosen day: p = {chosen:.2f}')
        ax.legend(); ax.grid(alpha=.2)
        matrix = np.array([[tn, fp], [fn, tp]])
        matrix_ax.imshow(matrix, cmap='Blues', vmin=0, vmax=len(y_test))
        for row in range(2):
            for col in range(2):
                matrix_ax.text(col, row, f'{[["TN", "FP"], ["FN", "TP"]][row][col]}\n{matrix[row, col]}',
                               ha='center', va='center', fontsize=13)
        matrix_ax.set(xticks=[0, 1], xticklabels=['No', 'Yes'], yticks=[0, 1], yticklabels=['No', 'Yes'],
                      xlabel='Predicted sellout', ylabel='Actual sellout', title='Test days')
        plt.tight_layout(); plt.show()
        print(f'所选日期：{"预计售罄，建议备货" if chosen >= threshold else "预计不售罄"}。测试集漏判 {fn} 天，误报 {fp} 天。')

    play = widgets.Play(value=50, min=0, max=100, step=2, interval=650, description='Play')
    threshold = widgets.FloatSlider(value=.5, min=0, max=1, step=.01, description='Threshold', continuous_update=False)
    # Play 是整数百分比；用 Python 回调连接到浮点门槛。
    def set_threshold(change):
        threshold.value = change['new'] / 100
    play.observe(set_threshold, names='value')
    def set_play(change):
        play.value = round(change['new'] * 100)
    threshold.observe(set_play, names='value')
    ad_budget = widgets.FloatSlider(value=5, min=0, max=10, step=.1, description='Ad budget', continuous_update=False)
    event = widgets.Dropdown(options=[('No event', 0), ('Weekend event', 1)], description='Event')
    output = widgets.interactive_output(explore_threshold, {'threshold': threshold, 'ad_budget': ad_budget, 'event': event})
    display(widgets.VBox([widgets.HBox([play, threshold]), ad_budget, event, output]))


def logistic_boundary(model_2d, X2_test, y2_test):
    def show_boundary(threshold):
        axis = np.linspace(-3.5, 3.5, 180)
        xx, yy = np.meshgrid(axis, axis)
        probability = model_2d.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1].reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.contourf(xx, yy, probability, levels=np.linspace(0, 1, 11), cmap='RdBu', alpha=.6)
        ax.contour(xx, yy, probability, levels=[threshold], colors='black', linewidths=2)
        ax.scatter(X2_test[:, 0], X2_test[:, 1], c=y2_test, cmap='RdBu', edgecolors='white', s=38)
        ax.set(xlabel='Feature 1', ylabel='Feature 2', title=f'Test points and threshold = {threshold:.2f}')
        fig.colorbar(image, ax=ax, label='Predicted probability')
        plt.show()

    boundary_control = widgets.interactive(show_boundary,
        threshold=widgets.FloatSlider(value=.5, min=.05, max=.95, step=.05, description='Threshold', continuous_update=False))
    display(boundary_control)


def kmeans_demo(X, kmeans_frames):
    def draw_kmeans(stage, k, seed):
        frames = kmeans_frames(X, k, seed)
        centers, labels, phase, previous, objective = frames[stage]
        fig, (ax, loss_ax) = plt.subplots(1, 2, figsize=(11, 4))
        if labels is None:
            ax.scatter(X[:, 0], X[:, 1], color='#aab4bf', s=30)
        else:
            ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', vmin=0, vmax=9, s=30, alpha=.8)
        ax.scatter(centers[:, 0], centers[:, 1], marker='*', s=250, c=range(k), cmap='tab10', vmin=0, vmax=9, edgecolors='black', linewidths=1)
        if previous is not None:
            for old, new in zip(previous, centers):
                ax.plot([old[0], new[0]], [old[1], new[1]], '--', color='#e05b42', lw=2)
        ax.set(title=f'{phase} · stage {stage}', xlabel='East–west position', ylabel='North–south position')
        ax.grid(alpha=.2)
        values = [(i, f[4]) for i, f in enumerate(frames[:stage+1]) if f[4] is not None]
        if values:
            loss_ax.plot(*zip(*values), '-o', color='#e05b42', markersize=4)
        loss_ax.set(xlim=(0, len(frames)-1), xlabel='Stage', ylabel='Sum of squared distances', title='Objective so far')
        loss_ax.grid(alpha=.2)
        fig.tight_layout(); plt.show()

    stage = widgets.IntSlider(value=0, min=0, max=16, description='阶段', continuous_update=False)
    play = widgets.Play(value=0, min=0, max=16, interval=650, description='播放')
    widgets.link((play, 'value'), (stage, 'value'))
    k_control = widgets.IntSlider(value=3, min=2, max=5, description='站点数', continuous_update=False)
    seed_control = widgets.IntSlider(value=2, min=0, max=9, description='种子', continuous_update=False)

    def reset_experiment(change):
        # 切换站点数或种子后，从新的初始化重新观察。
        play.playing = False
        stage.value = 0

    k_control.observe(reset_experiment, names='value')
    seed_control.observe(reset_experiment, names='value')
    out = widgets.interactive_output(draw_kmeans, {'stage': stage, 'k': k_control, 'seed': seed_control})
    display(widgets.VBox([widgets.HBox([play, stage]), k_control, seed_control, out]))


def hierarchy_demo(X):
    from scipy.cluster.hierarchy import linkage, dendrogram

    def membership_after_merges(Z, n, steps):
        # Z 每行记录要合并的两个节点；新节点编号依次是 n, n+1, ...
        nodes = {i: [i] for i in range(n)}
        active = set(range(n))
        last_pair = None
        for i, row in enumerate(Z[:steps]):
            a, b = int(row[0]), int(row[1])
            last_pair = (nodes[a], nodes[b])
            nodes[n + i] = nodes[a] + nodes[b]
            active.remove(a); active.remove(b); active.add(n + i)
        labels = np.empty(n, dtype=int)
        for color, node in enumerate(sorted(active)):
            labels[nodes[node]] = color
        return labels, last_pair

    def draw_hierarchy(step, method):
        Z = linkage(X, method=method)
        labels, pair = membership_after_merges(Z, len(X), step)
        fig, (ax, tree) = plt.subplots(1, 2, figsize=(12, 4))
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', vmin=0, vmax=9, s=100, edgecolors='white')
        for i, (x, y) in enumerate(X): ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        if pair:
            a, b = (X[group].mean(axis=0) for group in pair)
            ax.plot([a[0], b[0]], [a[1], b[1]], '--', color='#d4533c', lw=2)
        ax.set(title=f'{len(X)-step} clusters · merge {step}', xlabel='Size', ylabel='Sweetness'); ax.grid(alpha=.2)
        dendrogram(Z, ax=tree, color_threshold=0, above_threshold_color='#6c8190')
        if step: tree.axhline(Z[step-1, 2], color='#d4533c', linestyle='--', label='Current merge height'); tree.legend()
        tree.set(title=f'Dendrogram · {method}', xlabel='Fruit index', ylabel='Merge height')
        fig.tight_layout(); plt.show()

    step = widgets.IntSlider(value=0, min=0, max=len(X)-1, description='合并次数', continuous_update=False)
    play = widgets.Play(value=0, min=0, max=len(X)-1, interval=850, description='播放')
    widgets.link((play, 'value'), (step, 'value'))
    method = widgets.Dropdown(options=['single', 'complete', 'average', 'ward'], value='single', description='距离方式')
    out = widgets.interactive_output(draw_hierarchy, {'step': step, 'method': method})
    display(widgets.VBox([widgets.HBox([play, step]), method, out]))


def pca_projection(X2, mean2, centered2):
    def draw_projection(angle):
        theta = np.deg2rad(angle)
        direction = np.array([np.cos(theta), np.sin(theta)])
        score = centered2 @ direction
        reconstructed = mean2 + np.outer(score, direction)
        variance = np.var(score, ddof=1)
        mse = np.mean(np.sum((X2 - reconstructed)**2, axis=1))
        fig, (ax, chart) = plt.subplots(1, 2, figsize=(11, 4))
        ax.scatter(X2[:, 0], X2[:, 1], s=18, alpha=.55, color='#298d91', label='Original')
        ax.scatter(reconstructed[:, 0], reconstructed[:, 1], s=12, alpha=.5, color='#e05b42', label='Projection')
        for p, q in zip(X2[::4], reconstructed[::4]): ax.plot([p[0], q[0]], [p[1], q[1]], color='#98a6b0', lw=.7)
        line = mean2 + np.outer([-4, 4], direction)
        ax.plot(line[:, 0], line[:, 1], color='#e05b42', lw=2)
        ax.set(xlim=(-2, 8), ylim=(-2, 6), xlabel='Fruit width', ylabel='Fruit height', title=f'Projection · error {mse:.2f}')
        ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=.2)
        angles = np.arange(181)
        dirs = np.column_stack([np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))])
        variances = np.var(centered2 @ dirs.T, axis=0, ddof=1)
        chart.plot(angles, variances, color='#298d91'); chart.scatter([angle], [variance], color='#e05b42', s=70, zorder=3)
        chart.set(xlim=(0, 180), xlabel='Angle (degrees)', ylabel='Projected variance', title='How much variation survives?'); chart.grid(alpha=.2)
        fig.tight_layout(); plt.show()

    angle = widgets.IntSlider(value=0, min=0, max=180, step=2, description='角度', continuous_update=False)
    play = widgets.Play(value=0, min=0, max=180, step=6, interval=650, description='播放')
    widgets.link((play, 'value'), (angle, 'value'))
    out = widgets.interactive_output(draw_projection, {'angle': angle})
    display(widgets.VBox([widgets.HBox([play, angle]), out]))


def pca_digits(X_digits, pca_digits, digit_scores, cumulative, digits):
    def draw_digit(index, components):
        original = X_digits[index]
        restored = pca_digits.mean_ + digit_scores[index, :components] @ pca_digits.components_[:components]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
        axes[0].imshow(original.reshape(8, 8), cmap='gray', vmin=0, vmax=16)
        axes[0].set_title(f'Original · label {digits.target[index]}'); axes[0].axis('off')
        axes[1].imshow(restored.reshape(8, 8), cmap='gray', vmin=0, vmax=16)
        axes[1].set_title(f'Reconstructed · {components} components'); axes[1].axis('off')
        axes[2].plot(np.arange(1, 65), cumulative, color='#298d91')
        axes[2].scatter([components], [cumulative[components-1]], color='#e05b42', s=70)
        axes[2].set(xlabel='Components', ylabel='Cumulative explained variance', ylim=(0, 1.05), title='Dataset variation retained'); axes[2].grid(alpha=.2)
        fig.tight_layout(); plt.show()

    index = widgets.IntSlider(value=42, min=0, max=len(X_digits)-1, description='图片', continuous_update=False)
    components_widget = widgets.IntSlider(value=8, min=1, max=64, description='主成分', continuous_update=False)
    play = widgets.Play(value=8, min=1, max=64, step=2, interval=650, description='播放')
    widgets.link((play, 'value'), (components_widget, 'value'))
    out = widgets.interactive_output(draw_digit, {'index': index, 'components': components_widget})
    display(widgets.VBox([index, widgets.HBox([play, components_widget]), out]))


def q_update(update_q):
    def show_update(old, reward, next_best, done, alpha, gamma):
        target = reward if done else reward + gamma * next_best
        new = update_q(old, reward, next_best, done, alpha, gamma)
        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.barh(['old Q', 'target', 'new Q'], [old, target, new], color=['#94a3b8', '#f59e0b', '#2563eb'])
        ax.axvline(0, color='black', linewidth=.7)
        ax.set_title(f'TD error = {target-old:.2f}   |   update = {new-old:.2f}')
        ax.set_xlabel('value'); fig.tight_layout(); plt.show()

    widgets.interact(show_update,
        old=widgets.FloatSlider(value=2, min=-5, max=10, step=.5),
        reward=widgets.FloatSlider(value=-1, min=-5, max=10, step=1),
        next_best=widgets.FloatSlider(value=4, min=0, max=10, step=.5),
        done=False, alpha=widgets.FloatSlider(value=.5, min=0, max=1, step=.1),
        gamma=widgets.FloatSlider(value=.9, min=0, max=1, step=.1));


def q_training(train, SIZE, START, GOAL, SHELVES, ACTIONS):
    def draw_episode(history, episode):
        item = history[episode - 1]
        fig, (ax, score_ax) = plt.subplots(1, 2, figsize=(10, 4))
        board = np.zeros((SIZE, SIZE))
        for r, c in SHELVES: board[r, c] = 1
        ax.imshow(board, cmap='Greys', vmin=0, vmax=2)
        ax.set_xticks(range(SIZE)); ax.set_yticks(range(SIZE))
        ax.grid(color='#cbd5e1', linewidth=1)
        for r in range(SIZE):
            for c in range(SIZE):
                if (r, c) in SHELVES or (r, c) == GOAL: continue
                values = item['q'][r, c]
                if not np.any(values): continue  # 尚未更新，暂不画箭头
                a = int(np.argmax(values)); dr, dc = ACTIONS[a]
                ax.arrow(c, r, dc*.25, dr*.25, width=.025, head_width=.18, color='#2563eb', length_includes_head=True)
        path = np.array(item['path'])
        ax.plot(path[:, 1], path[:, 0], color='#f97316', alpha=.75, linewidth=2)
        ax.scatter([START[1], GOAL[1]], [START[0], GOAL[0]], c=['#10b981', '#ef4444'], s=150, zorder=4)
        ax.text(START[1], START[0], 'S', ha='center', va='center', color='white', zorder=5)
        ax.text(GOAL[1], GOAL[0], 'G', ha='center', va='center', color='white', zorder=5)
        ax.set_title('Blue: greedy action | Orange: episode path')
        returns = [x['return'] for x in history[:episode]]
        score_ax.plot(range(1, episode + 1), returns, color='#2563eb')
        score_ax.set(xlabel='episode', ylabel='total reward', title='Training returns (higher is better)')
        score_ax.axhline(0, color='#94a3b8', linewidth=.7)
        fig.tight_layout(); plt.show()
        status = '到达目标' if item['reached'] else '超时'
        print(f'第 {episode} 回合：{status}，走了 {len(item["path"])-1} 步，总奖励 {item["return"]:.0f}。箭头是本回合结束时的贪心选择；轨迹包含探索动作。')

    alpha_control = widgets.FloatSlider(value=.5, min=.1, max=1, step=.1, description='学习率 α')
    gamma_control = widgets.FloatSlider(value=.9, min=0, max=1, step=.1, description='折扣 γ')
    epsilon_control = widgets.FloatSlider(value=.2, min=0, max=1, step=.1, description='探索 ε')
    run_button = widgets.Button(description='重新训练', button_style='primary')
    play = widgets.Play(value=1, min=1, max=120, step=2, interval=650, description='播放')
    episode_slider = widgets.IntSlider(value=1, min=1, max=120, description='回合', continuous_update=False)
    widgets.link((play, 'value'), (episode_slider, 'value'))
    out = widgets.Output()
    state = {'history': None}

    def redraw(change=None):
        if state['history'] is not None:
            with out:
                clear_output(wait=True)
                draw_episode(state['history'], episode_slider.value)

    def rerun(_=None):
        state['history'] = train(alpha=alpha_control.value, gamma=gamma_control.value, epsilon=epsilon_control.value)
        episode_slider.value = 1
        redraw()

    episode_slider.observe(redraw, names='value')
    run_button.on_click(rerun)
    display(widgets.VBox([widgets.HBox([alpha_control, gamma_control, epsilon_control]), run_button,
                          widgets.HBox([play, episode_slider]), out]))
    rerun()
