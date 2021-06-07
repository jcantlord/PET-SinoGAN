# A GAN approach to synthetic PET imaging generation for breast cancer diagnosis

[![made-with-python](https://img.shields.io/badge/Coded%20with-Python-21496b.svg?style=for-the-badge&logo=Python)](https://www.python.org/)
[![made-with-latex](https://img.shields.io/badge/Documented%20with-LaTeX-4c9843.svg?style=for-the-badge&logo=Latex)](https://www.latex-project.org/)

Master's thesis; Universitat Oberta de Catalunya

**Author:** Javier Cantero Lorenzo

**Supervisor 1:** Héctor Espinós Morató

**Supervisor 2:** David Cascales Picó

# Experiments

DCGAN and WGAN with gradient penalty networks were tested for the synthesis of sinogram images with the aim of increasing the volume of training data for AI-based PET reconstructors in the context of breast cancer diagnosis.

ACRIN-6688 and ONCOVISION-MAMMI databases were the main case studies of this work.

The main evaluation metrics were manual inspection (Visual Turing Test) and the Fréchet Inception Distance score.

# Results

## Visual Turing Test

All evaluators confused a high percentage of generated images with real images.

![](figs/vtt.png)

## Quality vs Training

The quality of the generated images improved as the training progressed.

![](figs/qvt.png)

## Fréchet Inception Score

In all database-network pairs the FID score value decreased with training, which quantitatively supports that the models generate image sets with higher fidelity and greater diversity.

## Synthetic sinograms reconstructions

For each experiment, one of the generated sinograms was reconstructed by analytical methods (FBP), recovering the PET image with correct semantic representation of the anatomical region studied.

**DCGAN on ACRIN-6688**

![](figs/rec1.png)

**DCGAN on ONCOVISION-MAMMI**

![](figs/rec2.png)

**WGAN-GP on ONCOVISION-MAMMI**

![](figs/rec3.png)

## Conclusions

The studied networks are useful within the clinical context of PET medical imaging. The difference in the quality of the results between the two networks is not significantly different, which makes DCGAN preferable to WGAN-GP as it requires much less training time.
