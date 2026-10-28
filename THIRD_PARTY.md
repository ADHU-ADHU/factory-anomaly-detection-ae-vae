# Research and third-party references

- Thesis: Adhithya Palanisamy Chandrasekaran (2024), *Artificial Intelligence for Factory Automation – Anomaly Detection for Quality Control*, University of Skövde. [Permanent record](https://urn.kb.se/resolve?urn=urn:nbn:se:his:diva-24446). The README reproduces a published research chart with attribution; the architecture diagrams are redrawn from the implementation.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/), including their application-model implementations and pretrained weights, retain their respective terms and attribution requirements.
- [scikit-learn](https://scikit-learn.org/) supplies covariance, GMM and evaluation implementations in the portable adapter.
- The exploratory `ganomaly_ten.ipynb` invokes [chychen/tf2-ganomaly](https://github.com/chychen/tf2-ganomaly). This repository does not redistribute that upstream project or claim its implementation as original work.
- The experimental anomalib notebook uses [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib), a separate PyTorch/OpenVINO stack. It is not a dependency of the TensorFlow adapter.
- Tutorial/upstream URLs and attribution comments in archived notebook source are retained. Their inclusion does not imply that every scratch experiment is a completed original model.

The private industrial dataset, trained weights and notebook image outputs are not distributed. Refer to the original thesis and linked projects for their respective usage conditions; no blanket license over third-party material is asserted here.
