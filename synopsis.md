# SYNOPSIS: SOTA SENTIMENT ANALYSIS SYSTEM v4.0

**Name of the Student:** [Your Name]
**Student ID:** [Your Student ID]
**Class and Batch:** [Your Class/Batch]
**Roll no:** [Your Roll No]

---

### TITLE: 
**A Production-Grade Explainable Sentiment Analysis Ecosystem using DeBERTa-v3 and SHAP**

### INTRODUCTION:
This project focuses on the development of a state-of-the-art (SOTA) Sentiment Analysis ecosystem, progressing from baseline Transformer models (BERT) to highly optimized architectures like DeBERTa-v3. In the modern era of Natural Language Processing (NLP), understanding the "why" behind a model's prediction is as critical as the prediction itself. This study bridges the gap between high-performance inference and human-centric interpretability.

The system utilizes FastAPI for high-throughput serving and React for a dynamic, interactive dashboard. By implementing DeBERTa-v3, the model leverages Disentangled Attention and ELECTRA-style pre-training to achieve superior semantic understanding. The inclusion of SHAP (SHapley Additive exPlanations) ensures that the model is not a "black box," providing users with granular insights into which words influenced a specific sentiment score.

### OBJECTIVES:
- To implement a high-performance sentiment inference engine using DeBERTa-v3 optimized with BF16 and `torch.compile`.
- To integrate Explainable AI (XAI) using SHAP to provide token-level importance visualizations.
- To design a scalable API architecture using FastAPI with LRU caching for performance optimization.
- To develop a premium, responsive React-based frontend dashboard with real-time visualization of model confidence and explainability.

### MODULES:
1. **Core Inference Engine**: This module handles the loading of pre-trained SOTA models (DeBERTa-v3), tokenization cleaning, and hardware-accelerated inference (GPU/CPU fallback). It includes optimizations like dynamic padding and confidence calibration.
2. **Explainability (XAI) Suite**: A dedicated module that computes SHAP values or Attention-based importance weights for input text. It normalizes scores and maps them back to human-readable tokens for frontend consumption.
3. **Backend API Discovery & Orchestration**: Developed using FastAPI, this module serves as the bridge between the ML models and the user interface. It manages caching, validates input, and orchestrates the inference and explanation flows.
4. **Interactive Dashboard & Visualization**: The React-based frontend provides a seamless UX. It features "Batch Mode" for multiple reviews, individual "Explainer" cards, and high-fidelity visualizations of sentiment polarity using Glassmorphism design principles.

### MATERIAL AND METHODS:
- **Textual Data**: The IMDB Sentiment Dataset (HuggingFace) for training and validation.
- **Diagrams**: High-level system architecture showing the flow from raw text to FastAPI to Model to SHAP to UI.
- **Audio/Video**: Potential demonstration video showcasing the real-time speed of inference and the interactive nature of the SHAP tooltips in the dashboard.
- **Tools**: Python (PyTorch, Transformers, FastAPI, SHAP), JavaScript (React, Vite, CSS3).

### DURATION OF STUDY:
The study is estimated to take **4 weeks**:
- **Week 1 (Jan 22 - Jan 28)**: Literature review and model baseline implementation (BERT/RoBERTa).
- **Week 2 (Jan 29 - Feb 04)**: DeBERTa-v3 integration and inference optimization (BF16, Compile).
- **Week 3 (Feb 05 - Feb 11)**: SHAP Explainability module development and Backend API integration.
- **Week 4 (Feb 12 - Feb 18)**: Frontend Dashboard creation, full-system integration, and final verification.

### REFERENCES:
- **Data Collection Procedure**: Data is procedurally collected from the HuggingFace `datasets` library, specifically the `imdb` dataset, ensuring a balanced 50/50 split of positive and negative reviews for robust training.
- **Data Analysis Procedure**: Sentiment predictions are analyzed through accuracy metrics, F1-scores, and SHAP value distributions to ensure both quantitative performance and qualitative interpretability.
- **Literature**: 
  - 1. He, P., Liu, X., Gao, J., & Chen, W. (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*. arXiv preprint arXiv:2006.03654.
  - 2. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in neural information processing systems, 30.
  - 3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of deep bidirectional transformers for language understanding*. arXiv preprint arXiv:1810.04805.
  - 4. Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations.
  - 5. Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics.
