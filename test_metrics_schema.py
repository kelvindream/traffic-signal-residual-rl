"""C3 MetricsSchema：正式 CTDE 实验 CSV schema 单一来源的行为约束。"""

import tempfile
import unittest
from pathlib import Path

from utils.metrics_schema import (
    COMMON_COLUMNS,
    EPISODE_METRICS,
    EVAL_COLUMNS,
    MECHANISM_ZERO_ROW,
    PAIRED_COLUMNS,
    TRAIN_COLUMNS,
    write_csv,
)


class MetricsSchemaTests(unittest.TestCase):
    def test_column_groups_derive_from_common(self):
        self.assertEqual(
            COMMON_COLUMNS,
            ("scenario", "algo", "seed", "episode", "episode_seed",
             "total_reward", "decisions", "switches", "invalid_transitions")
            + EPISODE_METRICS)
        self.assertTrue(TRAIN_COLUMNS[:len(COMMON_COLUMNS)] == COMMON_COLUMNS)
        self.assertTrue(EVAL_COLUMNS[:len(COMMON_COLUMNS)] == COMMON_COLUMNS)
        # 训练列独有学习侧字段，评估列独有机制明细；两者均以 COMMON 为前缀
        self.assertNotIn("critic_loss", EVAL_COLUMNS)
        self.assertNotIn("learn_steps", EVAL_COLUMNS)
        self.assertIn("expected_intervention_rate", EVAL_COLUMNS)
        self.assertIn("expected_intervention_rate", TRAIN_COLUMNS)

    def test_paired_columns_cover_five_metrics(self):
        for metric in ("throughput", "unfinished", "co2_kg",
                       "time_average_queue_vehicles",
                       "total_waiting_increment_s"):
            self.assertIn(f"ctde_{metric}", PAIRED_COLUMNS)
            self.assertIn(f"fixed_cycle_{metric}", PAIRED_COLUMNS)
            self.assertIn(f"delta_{metric}", PAIRED_COLUMNS)
        self.assertIn("delta_total_reward", PAIRED_COLUMNS)
        self.assertIn("final_nonmask_residual_override_rate", PAIRED_COLUMNS)

    def test_mechanism_zero_row_matches_eval_mechanism_columns(self):
        mechanism_columns = set(EVAL_COLUMNS) - set(COMMON_COLUMNS)
        self.assertEqual(set(MECHANISM_ZERO_ROW), mechanism_columns)
        for key, value in MECHANISM_ZERO_ROW.items():
            self.assertEqual(value, 0 if key.endswith(("override", "veto",
                "eligible", "actions", "demand", "activity")) else 0.0)

    def test_write_csv_projects_rows_with_schema_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            write_csv(path, ("a", "b", "c"), [{"a": 1, "c": 2.5}])
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "a,b,c")
        self.assertEqual(lines[1], "1,,2.5")

    def test_write_csv_rejects_wrong_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            with self.assertRaises(ValueError):
                write_csv(path, ("a",), [{"a": 1, "b": 2}])


if __name__ == "__main__":
    unittest.main()
