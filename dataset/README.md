# WriteLens AI — Dataset Instructions

## Preferred Kaggle Dataset
For complete large-scale training, download the **AI vs Human Text Dataset** from Kaggle:
- **URL**: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
- **File Name**: `ai_vs_human.csv`
- **Labels**:
  - `0`: Human-written text
  - `1`: AI-generated text

Place the downloaded `ai_vs_human.csv` directly inside this `dataset/` directory before running:
```bash
python train.py
```

## Built-in Auto Dataset Generator
If `ai_vs_human.csv` is not present, `python train.py` will automatically generate an initial sample dataset so the model training pipeline and metrics generation can run out of the box without manual downloads!
