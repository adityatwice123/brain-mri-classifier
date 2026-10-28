# Brain Tumor MRI Classification

Explainable and lightweight classification of brain MRI slices into four classes
(glioma, meningioma, no tumor, pituitary) using transfer learning, with explainable AI,
model compression and an interactive Streamlit app.

**Live app:** [add your streamlit.app link here]

> Research prototype for a university project. It is not a medical device and must not be used for diagnosis.

Deep Learning Lab, Symbiosis Institute of Technology, Pune. Academic year 2026-27.

## What the project does

- Trains and compares three ImageNet-pretrained CNNs: MobileNetV2, EfficientNetB0 and ResNet50V2.
- Explains predictions with Grad-CAM, SHAP and LIME, and tests Grad-CAM with a deletion (faithfulness) test.
- Compresses the models with knowledge distillation, magnitude pruning and INT8 quantization.
- Includes pytest checks and a GitHub Actions workflow design.
- Serves everything in a ten-page Streamlit app, including a live prediction demo.

## Results (test set, 1,080 held-out images)

| Model | Parameters | Accuracy | Macro F1 | INT8 accuracy | INT8 size |
|---|---|---|---|---|---|
| MobileNetV2 (teacher) | 2.62M | 95.28% | 0.9526 | 78.24% | 3.08 MB |
| EfficientNetB0 | 4.42M | 95.74% | 0.9576 | 85.65% | 5.28 MB |
| ResNet50V2 | 24.13M | 96.85% | 0.9685 | 96.85% | 24.83 MB |
| MobileNetV2 student (distilled) | 2.34M | 91.67% | 0.9159 | 69.91% | 2.79 MB |

- A soft-vote ensemble of the three backbones reaches 97.50%.
- A small CNN trained from scratch reaches only 66.57%, so pretraining is what makes the project work.
- Only ResNet50V2's gain over MobileNetV2 is statistically significant (McNemar p = 0.028). EfficientNetB0 is not (p = 0.59).
- INT8 quantization badly hurt MobileNetV2 (95.28% to 78.24%) but not ResNet50V2.
- Grad-CAM did no better than random pixel removal in the deletion test (area 0.361 against 0.323), so the heatmaps are a rough guide only.

Training is not seeded, so a new run can differ by about one point.

## Dataset

A public four-class brain tumor MRI dataset, 7,200 images with 1,800 per class
(source: [add the dataset name and link here]). The Training and Testing folders were pooled
and re-split 70/15/15 (5,040 / 1,080 / 1,080), stratified, seed 42. Images are resized to
128 x 128 and scaled to 0 to 1.

## Model design

Every model is a pretrained backbone plus one shared head:
global average pooling, BatchNorm, Dense 256 (ReLU) with dropout 0.4, Dense 128 (ReLU) with
dropout 0.3, and a 4-way softmax. Training has two phases: the head on a frozen backbone
(Adam 1e-3), then the top 30 backbone layers unfrozen (Adam 1e-4).

## Repository structure

| Path | What it is |
|---|---|
| `app.py` | The Streamlit app (ten pages) |
| `content.py` | All numbers and text, taken from the notebook outputs |
| `requirements.txt` | Pinned dependencies |
| `assets/` | Figures exported from the notebook, diagrams and logo |
| `models/` | `teacher_model.keras` and `student_int8.tflite` |
| `samples/` | 16 sample MRI slices for the live demo |

## App pages

Overview, Dataset, Architecture and training, Results, Explainable AI, Compression, Live demo,
Testing and CI, Insights and limitations, About.

TensorFlow loads only when the Live demo page is opened, so all other pages open instantly.

## Run locally

```bash
git clone https://github.com/adityatwice123/brain-mri-classifier.git
cd brain-mri-classifier
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10 to 3.13 is needed for TensorFlow 2.21.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (main branch).
2. Go to https://share.streamlit.io and click **Create app**.
3. Choose this repository, branch `main`, main file `app.py`.
4. Under **Advanced settings** select **Python 3.12**, then click **Deploy**.

## Limitations

- One split and one seed, with no cross-validation. The test accuracy interval is about plus or minus 1 to 1.3 points.
- The original Training and Testing folders were pooled and re-split. The dataset contains 203 augmented meningioma images, so copies may sit on both sides of the split.
- "No tumor" slices are mostly axial T2 while tumor slices are mostly T1 contrast, so the model may partly learn scan type.
- 128-pixel input and no augmentation in training. No external test set.
- INT8 used only 100 calibration images. Pruning was run on a small separate CNN, not on the teacher. No student was trained without distillation.
- Model confidence is not accuracy. The app cannot tell whether an uploaded image is really a brain MRI.

## Team

| Name | PRN |
|---|---|
| Aditya Sharma | 24070126016 |
| Aadi Kulshresth | 24070126001 |
| Akriti Sahu | 24070126022 |

Under the guidance of Prof. Tanveer Ahmed.
