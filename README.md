# 👋 Hi, I'm Jackie NG
**Biochemist & AI Researcher | Bridging Sustainable Innovation and Healthcare Equity**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/yourprofile)  
[![ESG Startup](https://img.shields.io/badge/🌱-ESG%20Startup-green)](https://yourcompanywebsite.com)  
[![Twitter](https://img.shields.io/badge/Twitter-Follow-informational?style=flat&logo=twitter)](https://twitter.com/yourhandle)

---

## 🚀 Projects  

### 🩺 **MediVision**  
**Self-Supervised Interpretable Vision Transformer (ViT) for Early Disease Detection**  
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/medivision?style=social)](https://github.com/yourusername/medivision)  
[![Paper](https://img.shields.io/badge/📄-Preprint-orange)](https://arxiv.org/yourlink)  
[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/licenses/MIT)  

**Problem**: Over 75% of low-resource clinics lack tools for early diagnosis of lung cancer, diabetic retinopathy, and other critical diseases.  

**Solution**:  
- **Self-Supervised Pretraining**: Trained on 500k+ unlabeled medical images (CT scans, fundus photos) using masked autoencoding ([MAE](https://arxiv.org/abs/2111.06377)).  
- **Interpretable Attention**: Visualize ViT attention maps to highlight lesion regions (e.g., lung nodules, microaneurysms) for clinician trust.  
- **Low-Resource Deployment**: Optimized for <2GB VRAM GPUs; achieves **94% AUC-ROC** on NIH Chest X-ray (Top 5% benchmark).  

**Tech Stack**:  
```python
# Model Architecture
from vit import VisionTransformer
model = VisionTransformer(
    img_size=224, 
    patch_size=16, 
    embed_dim=768, 
    depth=12, 
    num_heads=12,
    qkv_bias=True,
    use_grad_cam=True  # Enable interpretability
)
