# SWELL HRV Synthetic Data Generation with Conditional TimeGAN

This repository implements a **Conditional TimeGAN** (Generative Adversarial Network) designed for generating synthetic Heart Rate Variability (HRV) time-series sequences. The generated data is conditioned on three distinct cognitive stress states, matching the classification tasks of the **SWELL-KW Dataset**.

---

## 📌 Project Overview
Stress detection through physiological markers (like HRV) is key to mental health monitoring. However, real-world physiological datasets are often limited due to privacy concerns and high collection costs. 

This project solves this by synthesizing sequential HRV data that preserves:
1. **Temporal Dependencies:** Sequential order is maintained (no random shuffling of time-steps).
2. **Conditional Context:** Generates features tailored to specific states: `no stress`, `interruption`, and `time pressure`.
3. **Multi-variate Correlations:** Correlates 35 HRV features (e.g., `MEAN_RR`, `SDRR`, `RMSSD`, `SDSD`) simultaneously.

---

## ⚙️ Model Architecture: Conditional TimeGAN
The model is composed of:
- **Embedder & Recovery Networks:** An autoencoder structure mapping the 35 HRV features into a lower-dimensional latent space.
- **Generator & Discriminator:** A GAN structure training on the latent representation to produce realistic sequence dynamics conditioned on the stress state.

---

## 📊 Key Results

### 1. GAN Stability & Training
The network reached a stable Nash equilibrium by Epoch 60, with the Generator loss settling around `0.80` and the Discriminator loss stabilizing near `1.35` (very close to the theoretical optimum where the discriminator cannot tell real data from synthetic).

### 2. Downstream Utility (TSTR)
To test if the synthetic data behaves like real data, we trained machine learning classifiers **exclusively on synthetic data** and tested them **exclusively on real test data** (Train on Synthetic, Test on Real):

| Model | Accuracy (Trained on Synthetic) | F1-Score |
| :--- | :---: | :---: |
| **MLP Neural Network** | **91.88%** | **0.9188** |
| **Random Forest** | **87.75%** | **0.8780** |
| **Random Forest (Real Baseline)** | *96.65%* | *0.9665* |

*The MLP trained on synthetic data successfully retains **~95%** of the real-data baseline performance, demonstrating excellent domain transferability.*

---

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm
   ```
2. Open and run the Jupyter notebook:
   [SWELL_Synthetic_Data_Generation_.ipynb](SWELL_Synthetic_Data_Generation_.ipynb) to train the model, save the outputs, and evaluate classifiers.

---

## 📄 License
This project is proprietary. All rights reserved by **Tharun Kumar T.**

No one is permitted to copy, distribute, modify, or create derivative works from this software without the express written permission of the copyright holder. Refer to the [LICENSE](LICENSE) file for more details.
