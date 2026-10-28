"""Regression checks for the notebook's kernel-driven animation controls."""
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np

from MachineLearning import teaching_widgets as demos


class KMeansAnimationTest(unittest.TestCase):
    def test_play_and_parameter_changes_drive_rendered_frames(self):
        data = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 3.], [4., 4.]])
        calls = []

        def frames(points, k, seed):
            calls.append((k, seed))
            centers = points[:k]
            labels = np.arange(len(points)) % k
            return [(centers, labels, f'frame {i}', None, float(16-i)) for i in range(17)]

        with patch.object(demos, 'display') as display, patch.object(demos.plt, 'show'):
            demos.kmeans_demo(data, frames)
            root = display.call_args.args[0]
            play, slider = root.children[0].children
            k_control, seed_control = root.children[1:3]

            for value in (1, 2, 16, 0):
                before = len(calls)
                play.value = value
                self.assertEqual(slider.value, value)
                self.assertGreater(len(calls), before, 'playback must recompute a frame')

            play.value = 8
            k_control.value = 4
            self.assertEqual((play.value, slider.value), (0, 0))
            self.assertEqual(calls[-1], (4, 2))
            play.value = 4
            seed_control.value = 3
            self.assertEqual((play.value, slider.value), (0, 0))
            self.assertEqual(calls[-1], (4, 3))

        demos.plt.close('all')


if __name__ == '__main__':
    unittest.main()
