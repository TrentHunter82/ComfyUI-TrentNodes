"""Boundary and cancellation tests; GPU inference is covered by the manual smoke workflow."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch

path = Path(__file__).resolve().parents[1] / 'nodes/eneas_segment.py'
spec = importlib.util.spec_from_file_location('eneas_segment_test', path)
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)


class EneasTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.zeros((2, 16, 24, 3))
        self.node = node.TrentEneasSegment()

    def test_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            self.node.segment(self.images, '../bad')

    def test_annotation_bounds(self):
        with self.assertRaisesRegex(ValueError, 'annotation_frame'):
            self.node.segment(self.images, 'track_text', annotation_frame=2)

    def test_point_bounds_and_labels(self):
        for points in ['[]', '[[24,1,1]]', '[[1,16,1]]', '[[1,1,2]]', '[[1,1]]']:
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.node.segment(self.images, 'track_points', points=points)

    def test_threshold_order(self):
        with self.assertRaisesRegex(ValueError, 'threshold'):
            self.node.segment(self.images, 'category', accept_threshold=0.1, reject_threshold=0.9)

    def test_cancel_kills_process_group_and_removes_frames(self):
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        interrupt = RuntimeError('cancelled')
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(node.Path, 'is_file', return_value=True), \
                patch.object(node.folder_paths, 'get_temp_directory', return_value=temp), \
                patch.object(node.mm, 'get_torch_device', return_value=torch.device('cuda:0')), \
                patch.object(node.mm, 'unload_all_models'), \
                patch.object(node.mm, 'soft_empty_cache'), \
                patch.object(node.mm, 'throw_exception_if_processing_interrupted', side_effect=[None, None, interrupt]), \
                patch.object(node.subprocess, 'Popen', return_value=process), \
                patch.object(node.os, 'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                self.node.segment(self.images, 'track_text')
            kill.assert_called_once_with(12345, node.signal.SIGTERM)
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
