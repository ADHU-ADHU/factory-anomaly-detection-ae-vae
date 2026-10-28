# Historical notebook index

This archive contains **36 original research notebooks** from the thesis code folder, organized by model family. Original filenames are retained, including spelling and capitalization. They record an evolving research process and are **not Run All ready**. Use the maintained TensorFlow package described in the [main README](../README.md) for the supported training and scoring entry points.

Notebook source was not executed during preparation. All saved outputs, execution counts, embedded attachments and arbitrary metadata were removed. Absolute Windows paths were replaced with `/path/to/private-data/...` placeholders; dataset/class folder names remain useful clues, while individual private image names become `example.bmp` or another matching extension. Relative model filenames and upstream attribution comments remain. No industrial dataset or pretrained thesis checkpoints are bundled.

[Source manifest](../notebooks/source-manifest.json) records original and sanitized SHA256 digests, categories, status and static validation results. All 36 notebooks pass nbformat validation. Ordinary code cells pass Python AST parsing; four cells containing IPython magics or shell commands are identified and skipped by the syntax check. **Syntax validity does not establish runtime correctness.**

## Autoencoders (17 notebooks)

| Notebook | Historical purpose |
|---|---|
| [Inlet_3_Left.ipynb](../notebooks/autoencoders/Inlet_3_Left.ipynb) | Custom CNN AE; inlet Type-1, left view. |
| [Inlet_3_right.ipynb](../notebooks/autoencoders/Inlet_3_right.ipynb) | Custom CNN AE; inlet Type-1, right view. |
| [Inlet_5_Left.ipynb](../notebooks/autoencoders/Inlet_5_Left.ipynb) | Custom CNN AE; inlet Type-2, left view, extended scoring. |
| [Inlet_5_Left_epoch50.ipynb](../notebooks/autoencoders/Inlet_5_Left_epoch50.ipynb) | Inlet Type-2 AE epoch variant. |
| [Inlet_5_Right.ipynb](../notebooks/autoencoders/Inlet_5_Right.ipynb) | Custom CNN AE; inlet Type-2, right view. |
| [Outlet_Left.ipynb](../notebooks/autoencoders/Outlet_Left.ipynb) | Custom CNN AE; outlet, left view, extended scoring. |
| [Outlet_right.ipynb](../notebooks/autoencoders/Outlet_right.ipynb) | Custom CNN AE; outlet, right view. |
| [Pretrained_in5_left_forgraph.ipynb](../notebooks/autoencoders/Pretrained_in5_left_forgraph.ipynb) | Inlet Type-2 AE scoring and visualization variant. |
| [Pretrained_in5_left_Inception.ipynb](../notebooks/autoencoders/Pretrained_in5_left_Inception.ipynb) | Inception encoder experiment; inlet Type-2. |
| [Pretrained_in5_left_InceptionResNetV2.ipynb](../notebooks/autoencoders/Pretrained_in5_left_InceptionResNetV2.ipynb) | InceptionResNetV2 AE; inlet Type-2. |
| [Pretrained_in5_left_InceptionResNetV2150epoc.ipynb](../notebooks/autoencoders/Pretrained_in5_left_InceptionResNetV2150epoc.ipynb) | InceptionResNetV2 AE; longer training variant. |
| [Pretrained_in5_left_Mobilenet_50epoc.ipynb](../notebooks/autoencoders/Pretrained_in5_left_Mobilenet_50epoc.ipynb) | MobileNetV2 AE; inlet Type-2. |
| [Pretrained_in5_left_Resnet50.ipynb](../notebooks/autoencoders/Pretrained_in5_left_Resnet50.ipynb) | ResNet50 AE; inlet Type-2. |
| [Pretrained_outlet_left_forgraph.ipynb](../notebooks/autoencoders/Pretrained_outlet_left_forgraph.ipynb) | Outlet AE scoring and visualization variant. |
| [Pretrained_outlet_left_InceptionResNetV2.ipynb](../notebooks/autoencoders/Pretrained_outlet_left_InceptionResNetV2.ipynb) | InceptionResNetV2 AE; outlet. |
| [Pretrained_Outlet_left_Mobilenet.ipynb](../notebooks/autoencoders/Pretrained_Outlet_left_Mobilenet.ipynb) | MobileNetV2 AE; outlet. |
| [Pretrained_Outlet_left_Resnet50.ipynb](../notebooks/autoencoders/Pretrained_Outlet_left_Resnet50.ipynb) | ResNet50 AE; outlet. |

## Variational autoencoders (10 notebooks)

| Notebook | Historical purpose |
|---|---|
| [alltogether_new_test_Variational_Autoencoders_.ipynb](../notebooks/variational_autoencoders/alltogether_new_test_Variational_Autoencoders_.ipynb) | Combined VAE exploration and multi-view evaluation. |
| [EPOCHchangevae.ipynb](../notebooks/variational_autoencoders/EPOCHchangevae.ipynb) | Epoch-sensitivity and VAE scoring experiment. |
| [graphforvae_inlet5.ipynb](../notebooks/variational_autoencoders/graphforvae_inlet5.ipynb) | Inlet Type-2 VAE scoring and visualization variant. |
| [New_VAE.ipynb](../notebooks/variational_autoencoders/New_VAE.ipynb) | Early VAE construction/training prototype. |
| [newtestvae.ipynb](../notebooks/variational_autoencoders/newtestvae.ipynb) | Additional VAE architecture/training prototype. |
| [Outlet_inceptionresnet_left_Variational_Autoencoders_.ipynb](../notebooks/variational_autoencoders/Outlet_inceptionresnet_left_Variational_Autoencoders_.ipynb) | InceptionResNetV2 VAE; outlet, extensive latent scoring. |
| [Outlet_left_Variational_Autoencoders_Implementation.ipynb](../notebooks/variational_autoencoders/Outlet_left_Variational_Autoencoders_Implementation.ipynb) | Custom CNN VAE; outlet. |
| [VAE.ipynb](../notebooks/variational_autoencoders/VAE.ipynb) | Early VAE architecture/training prototype. |
| [VAE_in5_left_inceptionresnet.ipynb](../notebooks/variational_autoencoders/VAE_in5_left_inceptionresnet.ipynb) | InceptionResNetV2 VAE; inlet Type-2. |
| [working_Variational_Autoencoders_Implementation.ipynb](../notebooks/variational_autoencoders/working_Variational_Autoencoders_Implementation.ipynb) | Custom CNN VAE; inlet Type-2. |

## Additional experiments (9 notebooks)

| Notebook | Historical purpose |
|---|---|
| [exception.ipynb](../notebooks/experiments/exception.ipynb) | Additional encoder/perceptual-loss exploration. |
| [GAN_anomaly detection.ipynb](../notebooks/experiments/GAN_anomaly%20detection.ipynb) | Incomplete dense TensorFlow GAN helper definitions. |
| [ganomaly_ten.ipynb](../notebooks/experiments/ganomaly_ten.ipynb) | Launcher for the third-party tf2-ganomaly repository. |
| [patchCNN_test.ipynb](../notebooks/experiments/patchCNN_test.ipynb) | Four-patch CNN encoder architecture experiment. |
| [Pretrained_featrue_extraction.ipynb](../notebooks/experiments/Pretrained_featrue_extraction.ipynb) | ResNet50 features with classical one-class estimators. |
| [Pretrained_model_autoencoder.ipynb](../notebooks/experiments/Pretrained_model_autoencoder.ipynb) | Incomplete pretrained AE construction/training prototype. |
| [PYtorch.ipynb](../notebooks/experiments/PYtorch.ipynb) | PyTorch ResNet18 AE and random-tensor shape demonstration. |
| [pytorchanomalib_GANomaly.ipynb](../notebooks/experiments/pytorchanomalib_GANomaly.ipynb) | GANomaly with PyTorch/anomalib and OpenVINO export. |
| [tensorflow_webgan.ipynb](../notebooks/experiments/tensorflow_webgan.ipynb) | TensorFlow DCGAN image-generation and score exploration. |

## Known blockers and interpretation

- **Private data and checkpoints:** placeholder paths intentionally do not resolve. Several notebooks save and later reload `.keras`, `.h5` or SavedModel artifacts that are not included. Dataset layout, preprocessing, class names and view selection must be checked before adapting an individual notebook.
- **Legacy TensorFlow/Keras:** source mixes `keras` and `tensorflow.keras`, directory generators, historical saving formats and custom VAE classes. API and serialization behavior changed across versions. A saved output in the original GAN notebook showed TensorFlow 2.10.1; the notebooks' inconsistent language-version metadata was removed rather than treated as an environment specification.
- **Evolving state:** repeated definitions, exploratory plots and references to variables from other cells can depend on execution order. Model filenames do not establish which experiment was the final thesis run. Positional layer lookups and Windows-specific `split("\\")` label handling also need review before reuse.
- **Evaluation choices:** some notebooks analyze latent means, others log variances or reconstructions. Threshold sweeps, covariance estimation, score direction and class composition must be checked when comparing results. Plot labels such as reconstruction “accuracy” are not a substitute for a held-out anomaly-detection evaluation.
- **Incomplete AE prototype:** `Pretrained_model_autoencoder.ipynb` references undefined `input_shape` and `autoencoder` names and has inconsistent validation resizing. It is retained as research history.
- **Incomplete GAN definitions:** `GAN_anomaly detection.ipynb` lacks the data/latent dimension assignments and NumPy import needed by its helpers; it does not run an end-to-end experiment.
- **External GANomaly launcher:** `ganomaly_ten.ipynb` preserves its reference to [chychen/tf2-ganomaly](https://github.com/chychen/tf2-ganomaly). It clones and invokes external code rather than supplying a complete original implementation. Its shell commands were preserved as text and never executed. A shell `cd` in one notebook cell also does not persist into a separate shell cell.
- **DCGAN scratch work:** `tensorflow_webgan.ipynb` repeats generator definitions, contains plotting cells that label discriminator scores as MSE, and ends with visualization inputs that are not defined in a clean execution. It is not a validated GAN anomaly-detection benchmark.
- **Separate framework:** `pytorchanomalib_GANomaly.ipynb` needs a compatible PyTorch/anomalib/OpenVINO stack with the historical APIs it uses. `PYtorch.ipynb` also uses PyTorch. Those dependencies are intentionally separate from the maintained TensorFlow package.

GAN literature and experiments complement the main AE/VAE study; their presence does not imply that a GAN model was part of the thesis's principal quantitative evaluation. See [thesis results](thesis-results.md) for the published comparisons and their limits.

## Recreate or verify the archive

From the repository root, with the original code directory available locally:

```bash
python tools/prepare_notebooks.py \
  --source-dir /path/to/original/only-codes \
  --output-dir notebooks \
  --validate
```

The sanitizer never executes cells or shell commands. `--validate` requires the optional `nbformat` package. The standalone greeting script `main.py` is omitted because it contains no thesis model implementation.

To verify an existing archive, its notebook structure and its sanitized SHA256 hashes:

```bash
python tools/prepare_notebooks.py --check notebooks
```
