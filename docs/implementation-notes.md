# From research notebooks to portable adapters

The archive and package serve different purposes. The notebooks document the original research and later experiments. The package provides explicit, testable model/data/scoring interfaces; it is not a claim that the old notebooks now reproduce their saved results.

## Preserved design

- RGB 400×320 inputs and latent dimension 512 by default.
- Six CNN convolution/pooling blocks, the original default transposed-convolution decoder, and cropping back to the input dimensions. The default ordinary AE has 3,956,451 parameters.
- Frozen application encoders. AE variants flatten features; VAE application variants use global average pooling.
- External RGB scaling to `[0,1]`. The original notebooks did not use application-specific `preprocess_input`; the adapter preserves that decision even when ImageNet weights are requested.
- VAE posterior mean/log variance, reparameterization during training, reconstruction objective and KL regularization.

## Changes made for reuse

| Area | Portable implementation |
| --- | --- |
| Paths and data | Explicit root directory, consistent resizing and separate normal train/validation/calibration splits |
| Keras models | Consistent `tf.keras`, instance-owned encoder/decoder, registered custom classes and `.keras` serialization |
| VAE objective | One matched MSE-plus-KL formula; no MSE/BCE mismatch or changing train/validation coefficient |
| Latent inference | Deterministic posterior mean, rather than sampled vectors or `mean + standard_deviation` |
| Score direction | MSE, `1 − SSIM`, Mahalanobis and negative GMM log likelihood all increase with anomaly |
| Covariance | Ledoit–Wolf shrinkage plus a positive numerical floor and a linear solve |
| Thresholds | Fit on a distinct normal calibration split, then evaluate once on test labels |
| Bookkeeping | Explicit histories, configuration and final-epoch encoder/decoder; separate best-validation checkpoint |
| Development shapes | Smaller image/latent dimensions supported; non-default decoder seed/cropping is adapted to the requested dimensions |

The portable VAE uses `reconstruction_weight × mean_pixel_MSE + mean_batch(sum_latent_KL)`. Its validation objective reconstructs posterior means; training samples the posterior. This deterministic validation estimator uses the same loss components and coefficient but is not the original Monte Carlo experiment.

## Historical notebook limitations

The archive contains Windows paths, absent checkpoint dependencies, interactive variable reuse and stale saved-result histories in the originals. Some notebook filenames imply 50/150 epochs while their actual code sets 125 or 2,000. Epoch settings in code take precedence over names.

Several VAE notebooks use global encoder/decoder variables, extensionless checkpoint paths or callback monitor names that do not match their logged loss. Others mix training/validation reconstruction formulas. Some evaluation experiments have score-sign or threshold-array inconsistencies, and select thresholds using the same anomaly examples on which they report F1. The adapter addresses these software issues through separate components and explicit calibration; it cannot retroactively validate historical numbers.

Historical scoring also explores KDE, ELBO-like scores, one-class SVM, cosine similarity and t-SNE. Those experiments remain in the notebooks; the CLI exposes the four scoring methods documented in setup. ResNet50, InceptionV3, PyTorch/anomalib and GAN prototypes are exploratory additions, not all completed comparative thesis models.

See [the notebook catalogue](notebook-index.md) for original filenames and known blockers. Shell/package installation cells in the archive are source material, not installation instructions for this package.

## Framework references

Compatibility work follows the [Keras 3 migration guide](https://keras.io/guides/migrating_to_keras_3/) and supported [ModelCheckpoint filenames](https://keras.io/api/callbacks/model_checkpoint/). Historical `ImageDataGenerator` is [deprecated](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator); the adapter uses `tf.data`. Official [InceptionResNetV2 preprocessing](https://keras.io/api/applications/inceptionresnetv2/inception_resnet_v2_model/) differs from the thesis scaling and should be evaluated as a documented alternative.
