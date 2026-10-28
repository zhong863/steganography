import unittest

import numpy as np

from grdh_eval.scoring import score
from grdh_eval.runner import summarize
from scripts.summarize_gpu_study import paired_bootstrap, score_means


class StudyReportTests(unittest.TestCase):
    def test_new_final_holdout_has_the_same_course_sample_requirement(self):
        rows = [dict(bpp=.0625, acc_jpeg=.9, acc_awgn=.8, acc_mblur=.95,
                     niqe=3.5, seconds=1) for _ in range(100)]
        self.assertTrue(summarize(rows, 100, "ecc_final")["course_sample_count_met"])
        self.assertFalse(summarize(rows[:20], 100, "ecc_final")["course_sample_count_met"])

    def test_vectorized_score_matches_scalar_clipping_and_formula(self):
        values = np.array([[.0625, .9, .9, .9, 4], [1, 1, .4, .8, 2], [2, .2, .7, .4, 9]])
        expected = [score(row[0], row[1:4], row[4]).total for row in values]
        np.testing.assert_allclose(score_means(values), expected, rtol=1e-12)

    def test_paired_bootstrap_zero_difference_and_identity_guard(self):
        rows = [dict(index=i, source_line=i + 12, bpp=.0625, acc_jpeg=.9,
                     acc_awgn=.8, acc_mblur=.95, niqe=3 + i / 10) for i in range(10)]
        result = paired_bootstrap(rows, rows, draws=100)
        self.assertEqual(result['difference'], 0)
        self.assertEqual(result['paired_bootstrap_95_interval'], [0, 0])
        with self.assertRaises(ValueError):
            paired_bootstrap(rows, list(reversed(rows)), draws=100)


if __name__ == '__main__':
    unittest.main()
