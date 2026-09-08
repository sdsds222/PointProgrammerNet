import unittest

import torch

from pointprogrammernet import PointProgrammerClassifier, PointProgrammerSegmenter


class PointProgrammerNetTest(unittest.TestCase):
    def test_frozen_parameter_counts_and_state_shapes(self):
        segmentation = PointProgrammerSegmenter()
        classification = PointProgrammerClassifier()
        self.assertEqual(sum(p.numel() for p in segmentation.parameters()), 366_389)
        self.assertEqual(sum(p.numel() for p in classification.parameters()), 529_426)
        points = torch.randn(2, 32, 3)
        for model in (segmentation, classification):
            self.assertEqual(
                [tuple(state.shape) for state in model.build_states(points)],
                [(2, 128, 170)] * 3,
            )

    def test_additive_merge(self):
        model = PointProgrammerSegmenter().eval()
        points = torch.randn(2, 96, 3)
        with torch.no_grad():
            direct = model.build_states(points)
            merged = model.merge_states(
                model.build_states(points[:, :37]),
                model.build_states(points[:, 37:]),
            )
        for left, right in zip(direct, merged):
            self.assertTrue(torch.allclose(left, right, rtol=2e-5, atol=2e-5))

    def test_forward_shapes(self):
        points = torch.randn(2, 64, 3)
        categories = torch.tensor([0, 4])
        self.assertEqual(PointProgrammerSegmenter()(points, categories).shape, (2, 64, 50))
        self.assertEqual(PointProgrammerClassifier()(points).shape, (2, 40))


if __name__ == "__main__":
    unittest.main()
