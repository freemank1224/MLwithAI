"""Check the teaching experiments' calculations and button-driven state."""
import unittest

from MachineLearning.watermelon_tree_lab import ManualTree, watermelon_tree_lab
from MachineLearning.watermelon_ensemble_labs import (
    BoostingState, ForestState, boosting_lab, forest_lab,
)


class TreeLabTest(unittest.TestCase):
    def test_tree_splits_prediction_and_reset(self):
        tree = ManualTree()
        self.assertAlmostEqual(tree.score((), 'color'), 4 / 9)
        self.assertAlmostEqual(tree.score((), 'texture'), .25)
        tree.split((), 'texture')
        self.assertIsNone(tree.score((False,), 'texture'))
        with self.assertRaises(ValueError):
            tree.split((False,), 'texture')
        tree.finish((False,))
        trace, good, total, closed = tree.predict(
            {'color': '乌黑', 'texture': '模糊', 'stem': '硬挺'})
        self.assertEqual((trace, good, total, closed), ([('texture', False)], 0, 2, True))
        tree.reset()
        self.assertEqual(tree.pending(), [()])

    def test_buttons_build_and_reset(self):
        root = watermelon_tree_lab()
        _, _, node, question, preview, buttons, _, tree_view, *rest = root.children
        split, finish, reset = buttons.children
        self.assertIn('0.250', preview.value)
        split.click()
        self.assertIn('纹理是清晰吗', tree_view.value)
        finish.click()
        self.assertIn('已定叶子', tree_view.value)
        reset.click()
        self.assertEqual(node.value, ())
        self.assertEqual(question.value, 'texture')
        self.assertNotIn('已定叶子', tree_view.value)

    def test_path_context_and_automatic_next_node(self):
        root = watermelon_tree_lab()
        _, _, node, question, preview, buttons, message, tree_view, *rest = root.children
        split, finish, reset = buttons.children

        split.click()  # 用纹理问题划分根节点
        labels = [label for label, _ in node.options]
        self.assertIn('纹理是清晰：是（瓜 A、B、C、D）', labels[0])
        self.assertIn('纹理是清晰：否（瓜 E、F）', labels[1])
        self.assertIn('还有 2 个待处理节点', message.value)
        self.assertIn('已自动切换到【根节点 → 纹理是清晰：是】', message.value)

        node.value = (False,)  # 学习者主动选择另一条分支
        self.assertIn('已手动切换到【根节点 → 纹理是清晰：否】', message.value)
        self.assertIn('当前节点：</b>根节点 → 纹理是清晰：否', preview.value)
        finish.click()
        self.assertIn('【根节点 → 纹理是清晰：否】中的瓜 E、F 定为叶子', message.value)
        self.assertIn('还有 1 个待处理节点', message.value)
        self.assertEqual(node.value, (True,))
        self.assertIn('已自动切换到【根节点 → 纹理是清晰：是】', message.value)

        question.value = 'stem'
        split.click()
        self.assertIn('纹理是清晰：是 → 瓜蒂是蜷缩：是', message.value)
        self.assertEqual(node.value, (True, True))
        finish.click()
        self.assertEqual(node.value, (True, False))
        finish.click()
        self.assertIn('所有节点都已处理，树已完成', message.value)
        self.assertTrue(node.disabled)
        self.assertTrue(split.disabled)
        self.assertTrue(finish.disabled)
        self.assertIsNone(node.value)
        self.assertNotIn('待处理节点', tree_view.value)

        reset.click()
        self.assertEqual(node.value, ())
        self.assertFalse(node.disabled)
        self.assertIn('待处理节点还剩 1 个', message.value)


class ForestLabTest(unittest.TestCase):
    def test_repeated_sampling_and_probability_aggregation(self):
        forest = ForestState()
        first = forest.add_tree('AABC', '纹理清晰吗？')
        self.assertFalse(first['did_split'])
        self.assertAlmostEqual(first['probability'], .75)
        second = forest.add_tree('ACEF', '纹理清晰吗？')
        self.assertTrue(second['did_split'])
        self.assertEqual(second['leaf'], ('A', 'C'))
        self.assertAlmostEqual(second['probability'], .5)
        forest.add_tree('EEEE', '纹理清晰吗？')
        self.assertAlmostEqual(forest.mean_probability(), (0.75 + 0.5) / 3)
        with self.assertRaises(ValueError):
            forest.add_tree('AAAA', '纹理清晰吗？')
        forest.reset()
        self.assertIsNone(forest.mean_probability())

    def test_buttons_add_and_reset(self):
        root = forest_lab()
        _, _, _, buttons, status, result = root.children
        add, reset = buttons.children
        self.assertIn('还没有树', result.value)
        add.click()
        self.assertIn('已建 1 棵树', status.value)
        self.assertIn('森林平均好瓜概率', result.value)
        reset.click()
        self.assertIn('还没有树', result.value)


class BoostingLabTest(unittest.TestCase):
    def test_two_rounds_use_updated_residuals(self):
        boost = BoostingState()
        first = boost.add_round(.35, .5)
        self.assertEqual(first['residual'], (-2., 0., 2.))
        self.assertEqual(first['adjustment'], (-2., 1., 1.))
        self.assertEqual(first['after'], (7., 8.5, 8.5))
        second = boost.add_round(.65, .5)
        self.assertEqual(second['residual'], (-1., -.5, 1.5))
        self.assertEqual(second['after'], (6.625, 8.125, 9.25))
        boost.reset()
        self.assertEqual(boost.prediction, (8., 8., 8.))
        self.assertEqual(boost.rounds, [])

    def test_buttons_add_and_reset(self):
        root = boosting_lab()
        _, current, _, _, _, buttons, status, history = root.children
        add, reset = buttons.children
        self.assertIn('8，8，8', current.value)
        add.click()
        self.assertIn('第 1 轮', status.value)
        self.assertIn('7，8.5，8.5', history.value)
        reset.click()
        self.assertIn('8，8，8', current.value)
        self.assertIn('还没有添加树', history.value)


if __name__ == '__main__':
    unittest.main()
