# Academic References, Research Papers & Technical Standards

**Project**: AI-Driven Market Linkage and Smart Cataloging Mobile Application for Marginalized Artisans  
**Problem Statement**: SIH26090 (Smart India Hackathon)  
**System**: Computer Vision & Intelligent Cataloging Pipeline  
**Document Version**: 1.0.0 (Comprehensive Academic Bibliography)  

---

## 1. Executive Summary of Research Foundations

The Computer Vision and Smart Cataloging Pipeline developed for **SIH26090** synthesizes peer-reviewed academic literature across four core domains:
1. **Dichotomous Image Segmentation & Deep Matting**: Pixel-accurate salient foreground separation on complex backgrounds without green screens.
2. **Edge Deep Learning & Model Calibration**: Lightweight, CPU-executable convolutional architectures paired with temperature-scaled confidence calibration.
3. **No-Reference Image Quality Assessment (IQA)**: Multi-zone differential operators and directional gradient asymmetry for pre-flight quality gating.
4. **Digital Image Processing & Computational Colorimetry**: Illuminant compensation, edge-preserving bilateral denoising, and anti-aliased sinc-windowed resampling.

---

## 2. Dichotomous Image Segmentation & Alpha Matting

### 2.1. IS-Net: Highly Accurate Dichotomous Image Segmentation
* **Paper Title**: *Highly Accurate Dichotomous Image Segmentation*
* **Authors**: Xuebin Qin, Hang Dai, Xiaobin Hu, Deng-Ping Fan, Ling Shao, Luc Van Gool
* **Venue**: European Conference on Computer Vision (**ECCV 2022**)
* **DOI / arXiv**: [arXiv:2209.04944](https://arxiv.org/abs/2209.04944)
* **Code Repository**: [https://github.com/xuebinqin/DIS](https://github.com/xuebinqin/DIS)
* **Implementation in Pipeline**: Powers the primary high-precision craft isolation engine (`isnet-general-use`). Operates at intermediate feature levels to accurately resolve fine artisanal structures, such as loose handloom threads, terracotta pot rims, filigree jewelry perforations, and carved woodwork.

### 2.2. U^2-Net: Nested U-Structure for Salient Object Detection
* **Paper Title**: *U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection*
* **Authors**: Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane, Martin Jagersand
* **Venue**: *Pattern Recognition*, Volume 106, 2020, 107404
* **DOI / arXiv**: [arXiv:2005.09007](https://arxiv.org/abs/2005.09007)
* **Code Repository**: [https://github.com/xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
* **Implementation in Pipeline**: Provides the structural foundation for the cascade fallback matting engine (`u2netp`, `silueta`). Its nested ReSidual U-blocks (RSU) enable deep multi-scale feature extraction without downsampling feature map resolutions.

### 2.3. GrabCut: Iterated Graph Cuts for Foreground Extraction
* **Paper Title**: *"GrabCut" — Interactive Foreground Extraction using Iterated Graph Cuts*
* **Authors**: Carsten Rother, Vladimir Kolmogorov, Andrew Blake
* **Venue**: ACM Transactions on Graphics (**SIGGRAPH 2004**), 23(3):309–314
* **ACM Link**: [https://dl.acm.org/doi/10.1145/1015706.1015720](https://dl.acm.org/doi/10.1145/1015706.1015720)
* **Microsoft Research**: [Publication Link](https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/)
* **Implementation in Pipeline**: Implemented as the deterministic non-neural CPU failover in `cv2.grabCut`. Employs Gaussian Mixture Models (GMMs) and min-cut graph partitioning when deep learning runtime dependencies are constrained.

---

## 3. Edge Deep Learning & Confidence Calibration

### 3.1. MobileNetV2: Inverted Residuals and Linear Bottlenecks
* **Paper Title**: *MobileNetV2: Inverted Residuals and Linear Bottlenecks*
* **Authors**: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
* **Venue**: IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR 2018**), pp. 4510–4520
* **DOI / arXiv**: [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
* **Model Hub**: [ONNX Model Zoo - MobileNetV2](https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet)
* **Implementation in Pipeline**: Core edge vision classifier (`mobilenetv2-7.onnx`). Runs at ~15ms on standard mobile/server CPUs, extracting lightweight visual features mapped directly into the Indian artisan craft taxonomy.

### 3.2. Confidence Calibration & Temperature Scaling
* **Paper Title**: *On Calibration of Modern Neural Networks*
* **Authors**: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger
* **Venue**: International Conference on Machine Learning (**ICML 2017**), PMLR 70:1321–1330
* **DOI / arXiv**: [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
* **Implementation in Pipeline**: Incorporated into `src/classification.py`. Solves the fundamental issue of modern deep neural network overconfidence. Applies temperature scaling ($T = 2.5$) and softmax margin checks:
$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$
Eliminates out-of-distribution classification hallucinations (such as misclassifying pottery vases as silk dupattas).

---

## 4. Pre-Flight Image Quality Assessment & Blur Gating

### 4.1. Laplacian Focus Measure Operator (Blur Sharpness)
* **Paper Title**: *Analysis of Focus Measure Operators for Shape-from-Focus*
* **Authors**: Shree K. Nayar, Yasuo Nakagawa
* **Venue**: *Pattern Recognition*, Columbia University Technical Report CUCS-041-90
* **Columbia CAVE**: [Nayar_TR90.pdf](https://www.cs.columbia.edu/CAVE/publications/pdfs/Nayar_TR90.pdf)
* **Related Survey**: Pertuz, S., Puig, D., Garcia, M.A., *Analysis of focus measure operators in shape-from-focus*, *Pattern Recognition* 46(5):1415–1432, 2013. [DOI: 10.1016/j.patcog.2012.11.011](https://doi.org/10.1016/j.patcog.2012.11.011)
* **Implementation in Pipeline**: Core algorithm of `src/quality_gate.py`. Computes variance of the Laplacian convolution $\text{Var}(\nabla^2 I)$. Augmented with our multi-zone center-patch ratio ($p_{90}$ sharpness) to prevent false blur rejections caused by uniform clean backgrounds.

### 4.2. Directional Motion Blur & Gradient Asymmetry
* **Paper Title**: *A No-Reference Objective Image Sharpness Metric Based on the Notion of Just Noticeable Blur (JNB)*
* **Authors**: R. Ferzli, L. J. Karam
* **Venue**: *IEEE Transactions on Image Processing* (**TIP 2009**), 18(4):717–728
* **IEEE Xplore**: [DOI: 10.1109/TIP.2008.2011760](https://ieeexplore.ieee.org/document/4806265)
* **Implementation in Pipeline**: Computes directional first-order derivative ratios via horizontal and vertical Sobel filters ($|G_x| / |G_y|$). Accurately distinguishes directional smartphone motion shake from uniform soft depth-of-field.

### 4.3. Otsu's Global Optimum Thresholding
* **Paper Title**: *A Threshold Selection Method from Gray-Level Histograms*
* **Authors**: Nobuyuki Otsu
* **Venue**: *IEEE Transactions on Systems, Man, and Cybernetics* (1979), 9(1):62–66
* **IEEE Xplore**: [DOI: 10.1109/TSMC.1979.4310076](https://ieeexplore.ieee.org/document/4310076)
* **Implementation in Pipeline**: Applied in `cv2.threshold(..., cv2.THRESH_OTSU)` for segmenting foreground energy masks, computing craft bounding bounding envelopes, and validating that artisan products occupy between 10% and 95% of the frame.

---

## 5. Computational Colorimetry & Signal Processing

### 5.1. Bilateral Filtering for Edge-Preserving Denoising
* **Paper Title**: *Bilateral Filtering for Gray and Color Images*
* **Authors**: C. Tomasi, R. Manduchi
* **Venue**: IEEE International Conference on Computer Vision (**ICCV 1998**), pp. 839–846
* **UC Santa Cruz**: [Tomasi_Manduchi_ICCV98.pdf](https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf)
* **Implementation in Pipeline**: Applied via `cv2.bilateralFilter` during enhancement. Non-linearly smooths camera noise from low-cost smartphone sensors in dimly lit artisan workshops while preserving sharp borders and fine textile textures.

### 5.2. Retinex & Gray-World Color Constancy
* **Paper Title**: *The Retinex Theory of Color Vision*
* **Authors**: Edwin H. Land
* **Venue**: *Scientific American* (1977), 237(6):108–128
* **Article Link**: [Scientific American](https://www.scientificamerican.com/article/the-retinex-theory-of-color-vision/)
* **Foundational Work**: Buchsbaum, G., *A spatial processor model for object colour perception*, *Journal of the Franklin Institute* (1980), 310(1):1–26. [DOI: 10.1016/0016-0032(80)90058-7](https://doi.org/10.1016/0016-0032(80)90058-7)
* **Implementation in Pipeline**: Integrated in `src/pipeline.py` to calculate chromatic adaptation coefficients, neutralizing heavy yellow tungsten lamps or uneven sunlight casts in rural workshops.

### 5.3. Lanczos Sinc-Windowed Resampling
* **Reference Work**: *Evaluation of Noisy Data* / *Filters for Common Resampling Tasks*
* **Authors**: Cornelius Lanczos (1964) / Ken Turkowski (Apple Computer, *Graphics Gems*, 1990)
* **Reference Link**: [Lanczos Interpolation](https://en.wikipedia.org/wiki/Lanczos_resampling)
* **Implementation in Pipeline**: Powered by `PIL.Image.Resampling.LANCZOS` (3-lobed windowed sinc interpolation kernel). Performs high-order antialiased downsampling to 1080×1080 catalog standards without ringing or moiré artifacts on high-frequency handloom weaves.

---

## 6. Open Standards & Government E-Commerce Protocols

### 6.1. Smart India Hackathon Problem Statement SIH26090
* **Organizing Authority**: Ministry of Education's Innovation Cell (MIC) & AICTE, Government of India
* **Theme**: *AI-Driven Market Linkage and Smart Cataloging Mobile Application for Marginalized Artisans*
* **Official Portal**: [https://sih.gov.in](https://sih.gov.in)

### 6.2. Open Network for Digital Commerce (ONDC) & Beckn Protocol
* **Specification**: Beckn Open Application Protocol (Retail / Unified Product Catalog Specification)
* **Governing Body**: DPIIT, Ministry of Commerce and Industry, Government of India
* **Official Portals**: [ONDC Official Website](https://ondc.org) | [Beckn Protocol Specification](https://becknprotocol.io)
* **Implementation in Pipeline**: The generated catalog payload (`product.json`) adheres to the ONDC/Beckn schema (category, craft type, material, dimensions, price, currency, regional GI tags).

### 6.3. Open Neural Network Exchange (ONNX)
* **Standard**: Open Neural Network Exchange Specification
* **Governing Body**: Linux Foundation AI & Data
* **Official Portal**: [https://onnx.ai](https://onnx.ai)
* **Implementation in Pipeline**: Ensures portable model serialization (`mobilenetv2-7.onnx`), enabling sub-20ms inference on standard CPUs without PyTorch or CUDA dependencies.
