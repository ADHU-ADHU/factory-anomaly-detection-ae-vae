"""Synthetic-only checks for score direction, split handling, and CLI workflows.

The tiny generated images exercise software integration. Their evaluation
numbers are not thesis results or an estimate of industrial model performance.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# Keep the smoke workflow bounded on machines with large CPU core counts.
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
import tensorflow as tf

from factory_anomaly.data import image_dataset, image_files
from factory_anomaly.scoring import (
    LatentScorer,
    calibrate_threshold,
    evaluation_metrics,
)
from factory_anomaly import vae as _vae  # Register custom checkpoint classes.


class ScoringTests(unittest.TestCase):
    def test_mahalanobis_handles_constant_and_rank_deficient_normal_latents(self):
        # More latent dimensions than examples, with no variance at all.
        constant = np.zeros((4, 8))
        scorer = LatentScorer("mahalanobis").fit(constant)
        scores = scorer.score(np.array([np.zeros(8), np.full(8, 2.0)]))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[1], scores[0])

        # A collinear normal population has singular sample covariance too.
        normal = np.repeat(np.arange(4, dtype=float)[:, None], 8, axis=1)
        scorer = LatentScorer("mahalanobis").fit(normal)
        scores = scorer.score(np.array([normal.mean(axis=0), np.full(8, 20.0)]))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[1], scores[0])

    def test_gmm_uses_negative_log_likelihood_with_higher_anomaly_scores(self):
        normal = np.array([[-0.02, 0.0], [0.0, -0.02], [0.02, 0.0], [0.0, 0.02]])
        scorer = LatentScorer("gmm", components=1, seed=7).fit(normal)
        centre_score, distant_score = scorer.score([[0.0, 0.0], [3.0, 3.0]])
        # A narrow Gaussian has density > 1 at its centre, hence negative NLL.
        self.assertLess(centre_score, 0.0)
        self.assertGreater(distant_score, 0.0)
        self.assertGreater(distant_score, centre_score)

    def test_normal_only_quantile_calibration_and_held_out_metrics(self):
        threshold = calibrate_threshold([1.0, 2.0, 3.0], quantile=0.5)
        self.assertEqual(threshold, 2.0)
        report = evaluation_metrics([0, 0, 1, 1], [0.5, 2.0, 5.0, 8.0], threshold)
        self.assertEqual(report["normal_count"], 2)
        self.assertEqual(report["anomaly_count"], 2)
        self.assertEqual(report["threshold"], threshold)
        for key in ("roc_auc", "average_precision", "precision", "recall", "f1"):
            self.assertEqual(report[key], 1.0, key)

    def test_invalid_fit_calibration_and_evaluation_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Fit the scorer"):
            LatentScorer().score([[1.0, 2.0]])
        for values in ([], [[1.0, 2.0]], [[np.nan, 1.0], [0.0, 1.0]]):
            with self.subTest(latents=values):
                with self.assertRaises(ValueError):
                    LatentScorer().fit(values)
        with self.assertRaisesRegex(ValueError, "GMM components"):
            LatentScorer("gmm", components=3).fit([[0.0], [1.0]])
        for scores in ([], [np.nan], [[1.0, 2.0]]):
            with self.subTest(calibration=scores):
                with self.assertRaises(ValueError):
                    calibrate_threshold(scores)
        for quantile in (0.0, 1.0, float("nan")):
            with self.subTest(quantile=quantile):
                with self.assertRaises(ValueError):
                    calibrate_threshold([1.0, 2.0], quantile)
        with self.assertRaises(ValueError):
            evaluation_metrics([0, 0], [1.0, 2.0], threshold=1.5)
        with self.assertRaises(ValueError):
            evaluation_metrics([0, 1], [1.0, np.inf], threshold=1.5)


class DataTests(unittest.TestCase):
    def test_missing_and_empty_image_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "does not exist"):
                image_files(root / "missing")
            (root / "readme.txt").write_text("No image data here.")
            with self.assertRaisesRegex(ValueError, "No BMP/PNG/JPEG"):
                image_files(root)
            with self.assertRaisesRegex(ValueError, "cannot be empty"):
                image_dataset([], (64, 64, 3))
            with self.assertRaisesRegex(ValueError, "batch_size"):
                image_dataset([root / "missing.png"], (64, 64, 3), batch_size=0)

    def test_rgb_resize_scaling_and_autoencoder_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            Image.new("RGB", (12, 8), color=(255, 128, 0)).save(nested / "normal.PNG")
            files = image_files(root)
            self.assertEqual(files, [nested / "normal.PNG"])
            images, targets = next(iter(image_dataset(files, (64, 64, 3), targets=True)))
            self.assertEqual(tuple(images.shape), (1, 64, 64, 3))
            self.assertEqual(images.dtype, tf.float32)
            np.testing.assert_allclose(images.numpy()[0, 0, 0], [1.0, 128 / 255, 0.0])
            np.testing.assert_array_equal(images.numpy(), targets.numpy())


class CliWorkflowTests(unittest.TestCase):
    """Use disjoint synthetic files for training, calibration, and evaluation."""

    def setUp(self):
        tf.keras.backend.clear_session()

    def tearDown(self):
        tf.keras.backend.clear_session()

    @staticmethod
    def _make_splits(root):
        counts = {
            "train/normal": 4,
            "validation/normal": 2,
            "calibration/normal": 3,
            "test/normal": 2,
            "test/anomaly": 2,
        }
        random = np.random.default_rng(37)
        for split, count in counts.items():
            path = root / split
            path.mkdir(parents=True)
            for index in range(count):
                if split == "test/anomaly":
                    pixels = random.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                else:
                    # Slightly varied normal examples keep latent covariance
                    # fitting meaningful without importing any real dataset.
                    pixels = random.integers(24, 50, (64, 64, 3), dtype=np.uint8)
                    pixels[16:48, 16:48] += np.uint8(70 + index)
                Image.fromarray(pixels).save(path / f"synthetic-{index}.png")

    def _cli(self, *arguments):
        environment = os.environ.copy()
        source = Path(__file__).resolve().parents[1] / "src"
        environment["PYTHONPATH"] = str(source) + os.pathsep + environment.get("PYTHONPATH", "")
        command = [sys.executable, "-m", "factory_anomaly.cli", *map(str, arguments)]
        completed = subprocess.run(
            command, env=environment, text=True, capture_output=True, timeout=120
        )
        self.assertEqual(
            completed.returncode, 0,
            "Command failed: " + " ".join(command) + "\n" + completed.stdout + completed.stderr,
        )
        return completed

    def _exercise_model(self, model_name):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "synthetic-data"
            run = root / f"{model_name}-run"
            self._make_splits(data)
            self._cli(
                "train", "--data-root", data, "--output", run,
                "--model", model_name, "--height", 64, "--width", 64,
                "--latent-dim", 4, "--epochs", 1, "--batch-size", 2,
            )
            for name in ("best.keras", "model.keras", "encoder.keras", "decoder.keras"):
                self.assertGreater((run / name).stat().st_size, 0, name)
            history = json.loads((run / "history.json").read_text())
            self.assertIn("loss", history)
            self.assertIn("val_loss", history)
            self.assertTrue(all(np.isfinite(value).all() for value in history.values()))
            config = json.loads((run / "config.json").read_text())
            self.assertEqual(config["training_count"], 4)
            self.assertEqual(config["validation_count"], 2)
            self.assertEqual(config["weights"], "none")
            checkpoint = tf.keras.models.load_model(run / "best.keras", compile=False)
            reconstruction = checkpoint(tf.zeros((1, 64, 64, 3)), training=False)
            self.assertEqual(tuple(reconstruction.shape), (1, 64, 64, 3))
            self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(reconstruction))))

            for method in ("mse", "ssim", "mahalanobis", "gmm"):
                with self.subTest(model=model_name, method=method):
                    self._cli(
                        "evaluate", "--data-root", data, "--run", run,
                        "--method", method, "--batch-size", 2, "--components", 1,
                    )
                    report = json.loads((run / f"evaluation-{method}.json").read_text())
                    self.assertEqual(report["normal_count"], 2)
                    self.assertEqual(report["anomaly_count"], 2)
                    self.assertEqual(report["calibration_count"], 3)
                    self.assertEqual(report["method"], method)
                    self.assertEqual(report["score_direction"], "higher = more anomalous")
                    self.assertTrue(np.isfinite(report["threshold"]))
                    for metric in ("roc_auc", "average_precision", "precision", "recall", "f1"):
                        self.assertTrue(0.0 <= report[metric] <= 1.0, metric)
                    with (run / f"scores-{method}.csv").open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 4)
                    self.assertEqual([int(row["label"]) for row in rows], [0, 0, 1, 1])
                    for row in rows:
                        self.assertTrue(row["relative_path"].startswith("test/"))
                        self.assertTrue(np.isfinite(float(row["score"])))

    def test_ae_train_checkpoint_and_all_four_evaluation_methods(self):
        self._exercise_model("ae")

    def test_vae_train_checkpoint_and_all_four_evaluation_methods(self):
        self._exercise_model("vae")


if __name__ == "__main__":
    unittest.main()
