# Brain Tumor MRI Classification (Streamlit app)

Interactive companion to the Deep Learning Lab project (Symbiosis Institute of Technology, Pune).
It shows everything done in the notebook: dataset, architecture, training, results, explainable AI
(Grad-CAM, SHAP, LIME, deletion test), compression (distillation, pruning, INT8) and a live demo.

Research prototype. Not a medical device and not for diagnosis.

## Files

| Path | What it is |
|---|---|
| `app.py` | The Streamlit app (ten pages) |
| `content.py` | All numbers and text, taken from the notebook outputs |
| `requirements.txt` | Pinned dependencies |
| `assets/` | Figures exported from the notebook, diagrams and logo |
| `models/` | `teacher_model.keras` (MobileNetV2) and `student_int8.tflite` (distilled student, INT8) |
| `samples/` | 16 sample MRI slices for the live demo |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository (main branch).
2. Go to https://share.streamlit.io, click **Create app** and choose the repository.
3. Main file path: `app.py`. Open **Advanced settings** and choose **Python 3.12**.
4. Click **Deploy**. The first build takes 5 to 10 minutes because TensorFlow is installed.

TensorFlow is only loaded when the **Live demo** page is opened, so every other page opens instantly.

## Notes on the numbers

All results come from the documented notebook run (Brain_Cancer_Final_4_Class_Report.ipynb).
Training is not seeded, so a new run can differ by about one point. The exported model files may
therefore differ slightly from the reported accuracy.
