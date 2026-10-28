"""Train a portable adapter and evaluate against a separate calibration split."""

import argparse
import csv
import json
import math
from pathlib import Path


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train(args):
    import tensorflow as tf
    from .data import image_files, image_dataset
    from .models import build_autoencoder
    from .vae import build_vae

    if args.epochs < 1 or args.batch_size < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    output = Path(args.output)
    if output.exists() and not output.is_dir():
        raise ValueError("output must be a directory path")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory to keep earlier runs")
    root = Path(args.data_root)
    train_files = image_files(root / "train" / "normal")
    validation_files = image_files(root / "validation" / "normal")
    tf.keras.utils.set_random_seed(args.seed)
    shape = (args.height, args.width, 3)
    builder = build_vae if args.model == "vae" else build_autoencoder
    kwargs = dict(image_shape=shape, latent_dim=args.latent_dim, backbone=args.backbone,
                  weights=None if args.weights == "none" else "imagenet")
    if args.model == "vae":
        kwargs["reconstruction_weight"] = args.reconstruction_weight
    encoder, decoder, model = builder(**kwargs)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), **({"loss": "mse"} if args.model == "ae" else {}))
    training = image_dataset(train_files, shape, args.batch_size, training=True, seed=args.seed, targets=True)
    validation = image_dataset(validation_files, shape, args.batch_size, targets=True)
    output.mkdir(parents=True, exist_ok=True)
    model(tf.zeros((1, *shape)), training=False)
    history = model.fit(training, validation_data=validation, epochs=args.epochs,
                        callbacks=[tf.keras.callbacks.ModelCheckpoint(
                            str(output / "best.keras"), monitor="val_loss", save_best_only=True
                        )], verbose=2)
    # Save final-epoch encoder/decoder together; best.keras is a separate checkpoint.
    model.save(output / "model.keras")
    encoder.save(output / "encoder.keras")
    decoder.save(output / "decoder.keras")
    config = {"model": args.model, "image_shape": shape, "latent_dim": args.latent_dim,
              "backbone": args.backbone, "weights": args.weights,
              "reconstruction_weight": args.reconstruction_weight, "epochs": args.epochs,
              "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "seed": args.seed, "training_count": len(train_files),
              "validation_count": len(validation_files), "tensorflow": tf.__version__,
              "keras": tf.keras.__version__, "checkpoint_used_for_evaluation": "final epoch"}
    _write_json(output / "config.json", config)
    _write_json(output / "history.json", history.history)
    print(f"Saved final-epoch model and encoder/decoder to {output}")


def evaluate(args):
    import numpy as np
    import tensorflow as tf
    from .data import image_files, image_dataset
    from .scoring import LatentScorer, calibrate_threshold, evaluation_metrics
    from . import vae  # Register custom Keras classes before loading.

    root, run = Path(args.data_root), Path(args.run)
    config = json.loads((run / "config.json").read_text())
    shape = tuple(config["image_shape"])
    encoder = tf.keras.models.load_model(run / "encoder.keras", compile=False)
    decoder = tf.keras.models.load_model(run / "decoder.keras", compile=False)

    def features_and_reconstruction_scores(files):
        latents, scores = [], []
        for images in image_dataset(files, shape, args.batch_size):
            encoded = encoder(images, training=False)
            latent = encoded[0] if isinstance(encoded, (list, tuple)) else encoded
            latents.append(latent.numpy())
            if args.method in ("mse", "ssim"):
                reconstructed = decoder(latent, training=False)
                batch_scores = (1 - tf.image.ssim(images, reconstructed, max_val=1.0)
                                if args.method == "ssim" else
                                tf.reduce_mean(tf.square(images - reconstructed), axis=(1, 2, 3)))
                scores.append(batch_scores.numpy())
        return np.concatenate(latents), np.concatenate(scores) if scores else None

    calibration_files = image_files(root / "calibration" / "normal")
    normal_files = image_files(root / "test" / "normal")
    anomaly_files = image_files(root / "test" / "anomaly")
    calibration_latents, calibration_scores = features_and_reconstruction_scores(calibration_files)
    test_files = normal_files + anomaly_files
    test_latents, test_scores = features_and_reconstruction_scores(test_files)
    if args.method in ("mahalanobis", "gmm"):
        training_files = image_files(root / "train" / "normal")
        training_latents, _ = features_and_reconstruction_scores(training_files)
        scorer = LatentScorer(args.method, args.components, config["seed"]).fit(training_latents)
        calibration_scores = scorer.score(calibration_latents)
        test_scores = scorer.score(test_latents)
    threshold = calibrate_threshold(calibration_scores, args.quantile)
    labels = np.array([0] * len(normal_files) + [1] * len(anomaly_files))
    report = evaluation_metrics(labels, test_scores, threshold)
    report.update(method=args.method, calibration_count=len(calibration_files),
                  calibration_quantile=args.quantile, score_direction="higher = more anomalous",
                  latent_representation="posterior mean" if config["model"] == "vae" else "AE bottleneck",
                  model_epoch="final", gmm_components=args.components if args.method == "gmm" else None)
    _write_json(run / f"evaluation-{args.method}.json", report)
    with (run / f"scores-{args.method}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["relative_path", "label", "score", "predicted_anomaly"])
        for file, label, score in zip(test_files, labels, test_scores):
            writer.writerow([str(file.relative_to(root)), int(label), float(score), int(score > threshold)])
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train on normal images only")
    training.add_argument("--data-root", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--model", choices=["ae", "vae"], default="ae")
    training.add_argument("--backbone", choices=["cnn", "mobilenetv2", "inceptionresnetv2", "resnet50"], default="cnn")
    training.add_argument("--weights", choices=["none", "imagenet"], default="none")
    training.add_argument("--height", type=int, default=400)
    training.add_argument("--width", type=int, default=320)
    training.add_argument("--latent-dim", type=int, default=512)
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--learning-rate", type=float, default=0.0005)
    training.add_argument("--reconstruction-weight", type=float, default=500.0)
    training.add_argument("--seed", type=int, default=42)
    training.set_defaults(handler=train)
    evaluation = commands.add_parser("evaluate", help="Calibrate on normal images, evaluate on held-out labels")
    evaluation.add_argument("--data-root", required=True)
    evaluation.add_argument("--run", required=True)
    evaluation.add_argument("--method", choices=["mse", "ssim", "mahalanobis", "gmm"], default="mahalanobis")
    evaluation.add_argument("--quantile", type=float, default=0.99)
    evaluation.add_argument("--components", type=int, default=2)
    evaluation.add_argument("--batch-size", type=int, default=4)
    evaluation.set_defaults(handler=evaluate)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
