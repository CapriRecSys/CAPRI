import unittest
import numpy as np
from metrics.accuracy import precisionk, recallk, mapk, ndcgk
from metrics.beyoundAccuracy import listDiversity, novelty, catalogCoverage, personalization


class TestMetrics(unittest.TestCase):
    # -------------------- Accuracy Metrics ------------------
    # Precision@k
    def test_precision1(self):
        actual = [2, 4, 5, 10]
        recommended = [1, 2, 3, 4, 5]
        expected = 3. / 5  # 3 items (2, 4, 5) out of five are relevant
        calculated = precisionk(actual, recommended)
        self.assertEqual(calculated, expected)

    def test_precision2(self):
        actual = [2, 4, 5, 10]
        recommended = [10, 5, 2, 4, 3]
        expected = 4. / 5
        calculated = precisionk(actual, recommended)
        self.assertEqual(calculated, expected)

    def test_precision3(self):
        actual = [2, 4, 5, 10]
        recommended = [1, 3, 6, 7, 8]
        expected = 0.0  # The intersection of these two sets is null
        calculated = precisionk(actual, recommended)
        self.assertEqual(calculated, expected)

    # Recall@k
    def test_recall1(self):
        actual = [2, 4, 5, 10]
        recommended = [1, 2, 3, 4, 5]
        expected = 3. / 4
        calculated = recallk(actual, recommended)
        self.assertEqual(calculated, expected)

    def test_recall2(self):
        actual = [2, 4, 5, 10]
        recommended = [10, 5, 2, 4, 3]
        expected = 1.0
        calculated = recallk(actual, recommended)
        self.assertEqual(calculated, expected)

    def test_recall3(self):
        actual = [2, 4, 5, 10]
        recommended = [1, 3, 6, 7, 8]
        expected = 0.0
        calculated = recallk(actual, recommended)
        self.assertEqual(calculated, expected)

    # mAP@K
    def test_mapk1(self):
        actual = [2, 4, 5, 10]
        predicted = [1, 2, 3, 4, 5]
        expected = (1. / 2 + 2. / 4 + 3. / 5) / 4
        calculated = mapk(actual, predicted)  # k=10 as default
        self.assertEqual(calculated, expected)

    def test_mapk2(self):
        actual = [2, 4, 5, 10]
        predicted = [10, 5, 2, 4, 3]
        expected = 1.0
        calculated = mapk(actual, predicted)
        self.assertEqual(calculated, expected)

    def test_mapk3(self):
        actual = [2, 4, 5, 10]
        predicted = [1, 3, 6, 7, 8]
        expected = 0.0
        calculated = mapk(actual, predicted)
        self.assertEqual(calculated, expected)

    def test_mapk4(self):
        actual = [2, 4, 5, 10]
        predicted = [11, 12, 13, 14, 15, 16, 2, 4, 5, 10]
        expected = (1. / 7 + 2. / 8 + 3. / 9 + 4. / 10) / 4
        calculated = mapk(actual, predicted)
        self.assertEqual(calculated, expected)

    def test_mapk5(self):
        actual = [2, 4, 5, 10]
        predicted = [2, 11, 12, 13, 14, 15, 4, 5, 10, 16]
        expected = (1. + 2. / 7 + 3. / 8 + 4. / 9) / 4
        calculated = mapk(actual, predicted)
        self.assertEqual(calculated, expected)

    # NDCG
    def test_ndcgk1(self):
        actual = [2, 4, 5, 10]
        predicted = [6, 7, 3, 8, 9]
        dcg = (0 / np.log2(2)) + (0 / np.log2(3)) + (0 / np.log2(4)) + \
            (0 / np.log2(5)) + (0 / np.log2(6))
        idcg = (1 / np.log2(2)) + (1 / np.log2(3)) + \
            (1 / np.log2(4)) + (1 / np.log2(5))
        expected = dcg / idcg
        calculated = ndcgk(actual, predicted)
        self.assertAlmostEqual(calculated, expected, 2)

    def test_ndcgk2(self):
        actual = [2, 4, 5, 10]
        predicted = [1, 2, 3, 40, 50]
        dcg = (0 / np.log2(2)) + (1 / np.log2(3)) + (0 / np.log2(4)) + \
            (0 / np.log2(5)) + (0 / np.log2(6))
        idcg = (1 / np.log2(2)) + (1 / np.log2(3)) + \
            (1 / np.log2(4)) + (1 / np.log2(5))
        expected = dcg / idcg
        calculated = ndcgk(actual, predicted)
        self.assertAlmostEqual(calculated, expected, 2)

    # -------------------- Beyound-Accuracy Metrics ------------------
    # Diversity
    def test_diversity_correct(self):
        predicted = [0, 0, 0]
        itemsSimilarityMatrix = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 3, 1]])
        expected = 0.0
        calculated = listDiversity(predicted, itemsSimilarityMatrix)
        self.assertEqual(calculated, expected)

    # Novelty
    def test_novelty_correct(self):
        predicted = [1, 2, 3]
        pop = {1: 10, 2: 20, 3: 30}
        numberOfUsers = 100
        listLength = 10
        expected = 0.74
        calculated = novelty(predicted, pop, numberOfUsers, listLength)
        self.assertAlmostEqual(calculated, expected, 2)

    # Catalog Coverage
    def test_catalogCoverage_correct(self):
        predicted = [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'Z']]
        catalog = ['A', 'B', 'C', 'X', 'Y', 'Z']
        expected = 83.3
        calculated = catalogCoverage(predicted, catalog)
        self.assertAlmostEqual(calculated, expected, 1)

    # Personalization
    def test_personalization_correct(self):
        predicted = [['A', 'B', 'C', 'D'], [
            'A', 'B', 'C', 'X'], ['A', 'B', 'C', 'Z']]
        expected = 0.25
        calculated = personalization(predicted)
        self.assertEqual(calculated, expected)
