# Active Learning with Particle Swarm Optimization for Enhanced Skin Cancer Classification Utilizing Deep CNN Models

### 📌 Overview

Skin cancer is one of the most prevalent global health challenges, with millions of cases diagnosed each year. Early detection is crucial to improving survival rates, yet existing deep learning models for automated diagnosis often require large annotated datasets and high computational resources.

This repository provides the official implementation of our published study, where we propose an efficient skin cancer classification framework that integrates Active Learning (AL) with Particle Swarm Optimization (PSO) to reduce labeling costs and boost model performance.


### 🧪 Research Highlights

Active Learning (AL): Selects the most informative unlabeled instances for annotation, reducing labeling requirements.

Particle Swarm Optimization (PSO): Enhances the AL selection strategy by identifying the most relevant data points.

CNN Models: Multiple pre-trained CNN architectures were evaluated on the HAM10000 skin lesion dataset.

### Results:

    Achieved ~89.4% accuracy using only 40% of the labeled training data.

    Demonstrated significant improvement in accuracy and efficiency over traditional approaches.

    The Least Confidence AL strategy outperformed other methods.

### 📊 Dataset

    HAM10000 Dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.

### ⚙️ Methodology

  1. Data Preprocessing – Image resizing, normalization, and augmentation.

  2. Active Learning Loop – Iteratively selects uncertain samples for expert annotation.

  3. PSO Integration – Optimizes the selection process within AL.

  4. Model Training – CNN models (EfficientNet, ResNet, DenseNet, VGG, etc.) trained on refined datasets.

  5. Evaluation – Accuracy, efficiency, and label cost reduction were measured.


### 🚀 Results
| Model  |	Accuracy (%)	|Data Used |
|:--------|:--------------|:----------|
|EfficientNetV2M	| 89.4 |	40% |
|ResNet101	| 87.2 |	40% |
|DenseNet121	| 86.9	| 40% |

✅ The AL-PSO framework consistently reduced annotation costs while maintaining high classification accuracy.

### 🚀 Why It’s Important

  Cuts annotation costs in medical imaging

  Improves efficiency for real-world clinical use

  Demonstrates AI + optimization for healthcare

### 📈 Highlights

  Active Learning loop selects the most “uncertain” samples

  PSO optimizes which samples to annotate

  Efficient results with fewer labeled examples



*⚡ AI meets healthcare: faster, smarter, and more efficient cancer detection. *
