# Multi-Modal Garbage Classification ♻️

A deep learning system that classifies garbage into Green, Blue, Black, or Other bins using both images and text descriptions. Built with EfficientNetV2-S for vision, DistilBERT for language, and trained on HPC (Compute Canada).
Note that both Jupiter notebook and .py files are provided here. Use .py for training on HPC. 
## 📌 Overview
Proper waste classification supports recycling, reduces landfill usage, and advances environmental sustainability. This project combines computer vision and natural language processing to improve classification accuracy.  
* Image branch → Extracts features from object photos.  
* Text branch → Extracts semantic meaning from item descriptions.  
* Fusion layer → Combines both modalities for the final prediction.

## 📂 Dataset Structure
garbage_data/  
├── Train/  
│   ├── Green/  
│   ├── Blue/  
│   ├── Black/  
│   ├── Other/  
├── Validation/  
│   ├── Green/  
│   ├── Blue/  
│   ├── Black/  
│   ├── Other/  
└── Test/  
    ├── Green/  
    ├── Blue/  
    ├── Black/  
    ├── Other/  
* Images: One object per photo, centered.  
* Text descriptions: Derived from image filenames (underscores → spaces).

## 🛠 Features
* Multi-Modal Architecture — Vision + Language fusion for higher accuracy. * Transfer Learning — Pretrained EfficientNetV2-S & DistilBERT. * HPC-Ready — SLURM scripts for Compute Canada clusters. * Data Augmentation — Improves robustness to lighting, angles, and backgrounds. * Early Stopping — Avoids overfitting.

## ⚙️ HPC
For Compute Canada:  
* Note that you have to load these versions of CUDA and Python to be compatible specifically for Compute Canada clusters.
```bash
module load cuda/12.2
module load python/3.12
source ~/ENEL645_env/bin/activate
```
HPC SLURM Example:
(Note that these are just examples. For instance, you do not have to use 2 GPUs.)
```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2 
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --job-name=garbage_classification
#SBATCH --output=output_%j
module load cuda/12.2
module load python/3.12
source ~/ENEL645_env/bin/activate
python gclasswithtxmg_copy.py
```
Also note that for training on HPC, you have to download the pretrained model from their website and then upload it to the cluster that is because the nodes in Compute Canada cannot connect to the internet.
## 🔮 Future Work
* MixUp / CutMix augmentation.
* Label smoothing.
* Smaller CNN backbone for faster inference.
* Collect more diverse dataset samples.

## 📚 References
[1] City of Calgary. (n.d.). What goes where?. https://www.calgary.ca. https://www.calgary.ca/waste/
what-goes-where/default.html  
[2] Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., ... & Adam, H. (2019). Searching
for mobilenetv3. In Proceedings of the IEEE/CVF international conference on computer vision (pp.
1314-1324).  
[3] Yang, Z., Xia, Z., Yang, G., & Lv, Y. (2022). A Garbage Classification Method Based on a Small
Convolution Neural Network. Sustainability, 14(22), 14735. https://doi.org/10.3390/su142214735  
[4] Sun, Z., Yu, H., Song, X., Liu, R., Yang, Y., & Zhou, D. (2020). Mobilebert: a compact task-agnostic
bert for resource-limited devices. arXiv preprint arXiv:2004.02984.  
[5] Wang, C., Qin, J., Qu, C., Ran, X., Liu, C., & Chen, B. (2021). A smart municipal waste management
system based on deep-learning and Internet of Things. Waste Management, 135, 20-29.  
[6] Grandini, M., Bagli, E., & Visani, G. (2020). Metrics for Multi-Class Classification: An Overview.
arXiv preprint arXiv:2008.05756.  

## 👥 Authors
* **Ali Karimi**
* Mohammad Alhashem
* Yuhao Huang 
* Noureldin Amer
