# Setup and execution

## Environment

Use Python 3.10–3.12 in a virtual environment. The local validation used Python 3.12, TensorFlow CPU 2.16.2, Keras 3.15.1, NumPy 1.26.4 and scikit-learn 1.5.2 on Linux. The pinned environment is deliberately separate from the historical notebook environment. GPU/CUDA configuration is not validated here.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
factory-anomaly --help
```

On Windows, activate with `.venv\Scripts\activate`. To inspect the archived notebooks, install `python -m pip install -e '.[notebooks]'`; the core environment does not include every dependency of the historical experiments.

## Prepare data

Keep one chosen inlier class under each `normal` directory. Use distinct images for training, validation, calibration and test. Supported formats are BMP, PNG and JPEG; subdirectories are searched recursively. Every image is decoded as RGB, resized to the configured height/width and divided by 255.

```text
your-data/
  train/normal/
  validation/normal/
  calibration/normal/
  test/normal/
  test/anomaly/
```

Train uses only the first two directories. Evaluation additionally needs the calibration and both test classes; latent methods refit a normal distribution from `train/normal`. No labels from the test set select the threshold. Split by underlying part/capture session where appropriate, and keep augmented copies of one source image within the same split.

## Training

```bash
factory-anomaly train --data-root /path/to/your-data \
  --output artifacts/ae-cnn --model ae --epochs 50

factory-anomaly train --data-root /path/to/your-data \
  --output artifacts/vae-cnn --model vae --epochs 50
```

Defaults: 400×320 RGB, latent size 512, batch size 4, Adam learning rate 0.0005 and seed 42. VAE loss is `500 × mean_pixel_MSE + summed_latent_KL`, averaged over the batch. Change the reconstruction coefficient with `--reconstruction-weight`.

Transfer encoders are frozen. Request pretrained weights explicitly:

```bash
factory-anomaly train --data-root /path/to/your-data \
  --output artifacts/vae-inception --model vae \
  --backbone inceptionresnetv2 --weights imagenet
```

Available backbones: `cnn`, `mobilenetv2`, `inceptionresnetv2`, `resnet50`. `--weights none` avoids downloads but gives a randomly initialized frozen application encoder, so it is not a pretrained experiment. The legacy `[0,1]` preprocessing is retained; official application preprocessing is a possible separate experiment, not silently inserted.

Flattened transfer AEs can be very large at the original resolution (roughly 91–163 million parameters in reviewed variants). VAE application features use global average pooling. Full-resolution training may need substantial memory; reduce batch size or use a smaller adapter input for development.

Each run writes `model.keras`, `encoder.keras`, `decoder.keras`, `best.keras`, `history.json` and `config.json`. The component files and evaluation use the **final epoch**; `best.keras` is a separate validation-loss checkpoint. Use a new/empty output directory for each run.

## Scoring and evaluation

```bash
factory-anomaly evaluate --data-root /path/to/your-data \
  --run artifacts/vae-cnn --method mahalanobis --quantile 0.99

factory-anomaly evaluate --data-root /path/to/your-data \
  --run artifacts/vae-cnn --method gmm --components 2
```

Methods: MSE, `1 − SSIM`, shrinkage-regularized Mahalanobis distance, and negative GMM log likelihood. Larger scores always indicate a greater anomaly. AE features are bottleneck vectors; VAE features are posterior means. Density models fit normal training latents; the threshold is a quantile of separate normal calibration scores. GMM component count must not exceed training sample count. Quantile defaults to 0.99 and is a configuration choice, not an established optimal threshold.

Evaluation writes `evaluation-METHOD.json` and `scores-METHOD.csv` in the run directory. Metrics include ROC-AUC, average precision, and precision/recall/F1 at the calibration threshold. Average precision here is not necessarily identical to the trapezoidal PR-AUC in the thesis tables.

## Verification

Run the offline checks without data or pretrained downloads:

```bash
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=-1 python -m unittest discover -s tests -v
```

Tests use small synthetic images, exercise gradient updates and `.keras` round trips, validate score direction/calibration, and run the training/evaluation CLI. Default full-resolution model geometry is constructed and checked. These checks establish software behavior; they do not reproduce industrial thesis results.

The original notebooks are not supported as a single `Run All` workflow. Consult [their catalogue](notebook-index.md) before attempting to adapt one.
