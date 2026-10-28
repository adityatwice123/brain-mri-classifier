"""All numbers and text shown in the app.

Every value below was read from the saved outputs of the notebook
Brain_Cancer_Final_4_Class_Report.ipynb (the documented run), so the app matches the report.
"""

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
FOLDER_CODE = {"gl": 0, "me": 1, "no": 2, "pi": 3}   # file prefix -> class index (Te-gl_10.jpg -> glioma)

TEAM = [("Aditya Sharma", "24070126016"), ("Aadi Kulshresth", "24070126001"), ("Akriti Sahu", "24070126022")]
GUIDE = "Prof. Tanveer Ahmed"

# ------------------------------------------------------------------ dataset
SPLIT = [  # class, total, train, val, test
    ("Glioma", 1800, 1260, 270, 270),
    ("Meningioma", 1800, 1260, 270, 270),
    ("No Tumor", 1800, 1260, 270, 270),
    ("Pituitary", 1800, 1260, 270, 270),
]
N_TEST = 1080

PREPROCESS = [
    ("Label from folder name", "The class of an image is the name of the folder it sits in.", "Works for any folder layout, including Training/Testing splits."),
    ("Convert to RGB", "Grayscale-looking MRI becomes 3 channels.", "Pretrained networks expect 3 channels."),
    ("Resize to 128 x 128", "cv2.resize.", "Saves memory and time (the arrays alone are about 1.4 GB). It loses fine detail."),
    ("Scale to 0 to 1", "Divide by 255, float32.", "Stable training."),
    ("Backbone rescaling", "EfficientNetB0 multiplies by 255, ResNet50V2 maps to -1..1, MobileNetV2 uses 0..1 as supplied.", "Each pretrained backbone expects a different input range. A Rescaling layer inside the model handles it."),
    ("One-hot labels", "to_categorical with 4 classes.", "Needed for categorical cross-entropy."),
    ("Stratified 70/15/15 split", "Two train_test_split calls, stratify on the label, seed 42.", "Same class mix in every split, reproducible."),
    ("Augmentation", "Pipeline built (rotation 15 degrees, zoom 0.1, flip, brightness 0.8 to 1.2) and visualised, but NOT used in training.", "A known limitation."),
]

# ------------------------------------------------------------------ models
# Float accuracy, INT8 accuracy, latency etc. all come from the notebook (Phase 2B and Phase 4B tables).
MODELS = [
    dict(name="MobileNetV2 (teacher)", short="MobileNetV2", params=2.62, params_exact="2,624,452", acc=95.28, f1=0.9526,
         int8_acc=78.24, int8_f1=0.7550, f32_mb=10.50, int8_mb=3.08, keras_mb=26.18, latency=4.16, train_min=None, p=None,
         best_val_a=0.9435, best_val_b=0.9444, color="#FF9800"),
    dict(name="EfficientNetB0", short="EfficientNetB0", params=4.42, params_exact="4,416,039", acc=95.74, f1=0.9576,
         int8_acc=85.65, int8_f1=0.8534, f32_mb=17.66, int8_mb=5.28, keras_mb=33.22, latency=6.60, train_min=2.9, p=0.5896,
         best_val_a=0.9361, best_val_b=0.9509, color="#2196F3"),
    dict(name="ResNet50V2", short="ResNet50V2", params=24.13, params_exact="24,130,948", acc=96.85, f1=0.9685,
         int8_acc=96.85, int8_f1=0.9686, f32_mb=96.52, int8_mb=24.83, keras_mb=217.06, latency=36.37, train_min=4.9, p=0.0280,
         best_val_a=0.9296, best_val_b=0.9694, color="#2ECC71"),
    dict(name="MobileNetV2 student (KD)", short="Student (KD)", params=2.34, params_exact="2,340,228", acc=91.67, f1=0.9159,
         int8_acc=69.91, int8_f1=0.6562, f32_mb=9.36, int8_mb=2.79, keras_mb=9.89, latency=4.06, train_min=None, p=None,
         best_val_a=None, best_val_b=None, color="#9B59B6"),
]
ENSEMBLE = dict(acc=97.50, f1=0.9750)
SMALL_CNN_ACC = 66.57

# Test-set confusion matrices (rows = true class, columns = predicted), read from the notebook figures.
CONFUSION = {
    "MobileNetV2 (teacher)": [[257, 9, 3, 1], [15, 243, 3, 9], [3, 2, 264, 1], [2, 3, 0, 265]],
    "EfficientNetB0":        [[244, 22, 2, 2], [3, 264, 1, 2], [1, 3, 264, 2], [1, 7, 0, 262]],
    "ResNet50V2":            [[251, 14, 1, 4], [0, 267, 1, 2], [2, 2, 264, 2], [1, 3, 2, 264]],
}
MCNEMAR = [  # model, teacher-only correct, model-only correct, p
    ("EfficientNetB0", 25, 30, 0.5896),
    ("ResNet50V2", 18, 35, 0.0280),
]

MODEL_INFO = {
    "MobileNetV2 (teacher)": dict(
        what="A lightweight CNN built for phones and edge devices. It is the reference model and the 'teacher' of the project.",
        how=["Depthwise separable convolution: a 3x3 filter per channel, then a 1x1 convolution that mixes channels. About 8 to 9 times less computation than a normal convolution.",
             "Inverted residual block: expand with 1x1 (x6), filter with 3x3 depthwise, squeeze back with a linear 1x1. A skip connection is used when sizes match.",
             "At 128 pixels its final feature map is only 4 x 4."],
        result="95.28% accuracy, macro F1 0.9526. Weakest class is meningioma (recall 0.90, confused with glioma). Training reached 99.7% while validation stayed near 94.4%, so it overfits a little.",
        why="Pretrained features transfer well and it suits a 5,040-image training set. Its catch is quantization: INT8 dropped it to 78.24%."),
    "EfficientNetB0": dict(
        what="A CNN found by architecture search and scaled with one rule. B0 is the smallest member of the family.",
        how=["MBConv blocks (MobileNetV2 style) plus squeeze-and-excitation, which learns which channels matter and rescales them.",
             "Swish activation instead of ReLU.",
             "Compound scaling grows depth, width and input size together."],
        result="95.74% accuracy, macro F1 0.9576. McNemar p = 0.59 against MobileNetV2, so no real gain. Weak spot: glioma recall 0.904 (22 gliomas read as meningioma). INT8: 85.65%.",
        why="Likely untested reasons for no gain: it was designed for 224-pixel and larger inputs and we used 128, and this dataset is easy enough that extra sophistication adds little."),
    "ResNet50V2": dict(
        what="A deep CNN of 50 layers built around skip connections.",
        how=["A residual block outputs F(x) + x, so gradients flow through very deep networks.",
             "Bottleneck blocks (1x1 reduce, 3x3, 1x1 expand) repeat in stages of 3, 4, 6 and 3.",
             "'V2' means BatchNorm and ReLU come before each convolution, which eases optimisation. It ends in 2048 channels."],
        result="96.85% accuracy, macro F1 0.9685, the best single model. McNemar p = 0.028, a real gain. The most balanced class scores (F1 0.958, 0.960, 0.981, 0.974). The only model unharmed by INT8.",
        why="About 9 times the parameters of MobileNetV2, so more capacity. The price is a 24.83 MB INT8 file and about 36 ms per image, roughly 9 times slower, for 1.6 more points."),
    "MobileNetV2 student (KD)": dict(
        what="A smaller MobileNetV2 trained by knowledge distillation from the teacher.",
        how=["Only the last 10 backbone layers are trainable, with a 64-unit head (814,724 trainable parameters against 1,890,308 for the teacher).",
             "It learns from the teacher's softened probabilities and from the true labels.",
             "Loss = 0.7 x T^2 x KL(teacher soft, student soft) + 0.3 x cross-entropy(true label, student), with T = 4."],
        result="91.67% accuracy, macro F1 0.9159: 3.61 points below the teacher. After INT8 it falls to 69.91%.",
        why="Less trainable capacity. It is neither much smaller (2.34M against 2.62M parameters) nor faster (about 4 ms each), and no student was trained without distillation, so we cannot claim distillation helped."),
}

LAYERS = [
    ("Input", "MRI slice, 128 x 128 x 3, values 0 to 1", "Same preprocessing as training."),
    ("Rescaling", "MobileNetV2: none. EfficientNetB0: x255. ResNet50V2: x2 - 1", "Each backbone expects a different range."),
    ("Backbone", "Pretrained on ImageNet", "Early layers already detect edges and textures that help on MRI."),
    ("Global average pooling", "Each feature map becomes one number", "Far fewer parameters than Flatten, so less overfitting."),
    ("BatchNorm", "Normalises activations", "Stabilises training."),
    ("Dense 256 (ReLU) + Dropout 0.4", "First classifier layer", "Dropout switches off units at random so the network cannot memorise."),
    ("Dense 128 (ReLU) + Dropout 0.3", "Second classifier layer", "Same idea, smaller layer."),
    ("Dense 4 (softmax)", "Four probabilities that sum to 1", "One probability per class."),
]

TRAINING = [  # setting, phase A, phase B
    ("Trainable layers", "Head only, backbone frozen", "Head plus top 30 backbone layers"),
    ("Optimiser and learning rate", "Adam, 1e-3", "Adam, 1e-4"),
    ("Batch size", "64", "32"),
    ("Maximum epochs", "15", "15"),
    ("Early stopping (validation accuracy)", "Patience 5, best weights restored", "Patience 6, best weights restored"),
    ("Learning-rate reduction", "Factor 0.5, patience 3", "Factor 0.5, patience 3"),
    ("Loss", "Categorical cross-entropy", "Categorical cross-entropy"),
]

# ------------------------------------------------------------------ explainability
XAI = {
    "Grad-CAM": dict(
        how=["Take the last convolutional layer: it has the richest features and still keeps some position information.",
             "Compute the gradient of the predicted class score with respect to each feature map.",
             "Average each map's gradients over width and height to get one weight per channel.",
             "Add the feature maps up using those weights, keep positive influence (ReLU), resize to the image and overlay it."],
        found="On the teacher, heat sat on broad central or upper regions. On the meningioma slice it covered the mass, but on the other classes it did not clearly sit on a lesion. Different backbones highlighted different places while agreeing on the label.",
        caveat="At 128 pixels MobileNetV2's final map is only 4 x 4, so every heatmap is a coarse blur.",
        figs=[("gradcam_teacher.png", "Grad-CAM of the teacher, one test slice per class"),
              ("gradcam_backbones.png", "Grad-CAM of the three backbones on the same slices")]),
    "SHAP": dict(
        how=["Gives each pixel a contribution to the predicted class, based on Shapley values from game theory.",
             "We used a gradient explainer with 50 training images as background."],
        found="The maps are speckled with sensible magnitudes (about 0.006 to 0.02) but show no clear lesion-shaped region.",
        caveat="Pixel-level attributions are noisy on this model, so we draw no conclusion from them.",
        figs=[("shap.png", "SHAP attribution maps for the predicted class of the teacher")]),
    "LIME": dict(
        how=["Split the image into superpixels.",
             "Switch random groups of superpixels off (500 samples) and record the model output.",
             "Fit a simple linear model to see which superpixels drive the prediction. The method is random, so repeat runs can differ."],
        found="LIME highlighted large regions of skull and brain, and the regions it kept were not clearly tied to the tumor. Some slices carry stock-photo watermarks (visible in the no-tumor row).",
        caveat="Right column shows the superpixels LIME kept as a mask.",
        figs=[("lime.png", "LIME explanations: original, segments, kept regions")]),
    "Deletion test": dict(
        how=["Remove the pixels Grad-CAM ranks highest, in 10% steps, on 60 test images (15 per class).",
             "Track the model's confidence in its original prediction.",
             "Repeat with the same number of random pixels. A faithful heatmap should make confidence fall faster than random removal."],
        found="Area under the curve: 0.361 for Grad-CAM against 0.323 for random order. Grad-CAM was no better than random, so the heatmaps are weak evidence. A fully black image was still labelled 'no tumor' with high confidence.",
        caveat="Treat Grad-CAM as a rough guide to where the model looks, not proof of what it relies on.",
        figs=[("deletion_test.png", "Deletion (faithfulness) test, measured")]),
    "Teacher vs student maps": dict(
        how=["Compute Grad-CAM for the teacher and for the distilled student on the same slices and compare."],
        found="The maps often differ. On the glioma slice the teacher's heat sits at the bottom of the image while the student's sits near the centre. On the meningioma slice both cover the left side where the mass is, but not in the same way. Distillation did not make the student look at the same places as the teacher.",
        caveat="Both heatmaps are coarse and only a rough guide.",
        figs=[("gradcam_teacher_vs_student.png", "Grad-CAM of the teacher and the distilled student")]),
}

# ------------------------------------------------------------------ compression
PRUNING = [  # target, measured, accuracy, gzip MB
    ("0% (dense base)", "0.0%", 66.57, 0.41),
    ("50%", "49.9%", 85.93, 0.25),
    ("70%", "69.9%", 77.04, 0.17),
    ("90%", "89.9%", 50.00, 0.08),
]

PYTEST = [
    ("test_data.py", "test_class_count", "There are exactly 4 classes"),
    ("test_data.py", "test_class_names_unique_and_sorted", "Class names are unique and sorted"),
    ("test_data.py", "test_split_is_70_15_15", "Split sizes are about 70 / 15 / 15"),
    ("test_data.py", "test_sample_data_exists", "Every class has at least 3 sample images"),
    ("test_data.py", "test_image_shape", "Images resize to 128 x 128 x 3"),
    ("test_data.py", "test_pixel_normalization", "Scaled pixels stay within 0 to 1"),
    ("test_data.py", "test_no_nan_in_images", "No NaN values in the images"),
    ("test_model.py", "test_model_output_shape", "Model output has shape (1, 4)"),
    ("test_model.py", "test_model_probabilities_sum_to_one", "Probabilities sum to 1"),
    ("test_model.py", "test_tflite_output_shape", "TFLite model output has shape (1, 4)"),
    ("test_model.py", "test_model_size_under_limit", "INT8 file is under 10 MB"),
    ("test_model.py", "test_accuracy_threshold", "Small accuracy check on sample images"),
]

CI_YAML = """name: Brain Cancer MRI CI Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - run: pip install -r requirements.txt
    - run: pytest tests/test_data.py -v
    - run: pytest tests/test_model.py -v
    - run: pytest tests/ -v --tb=short
"""

INSIGHTS = [
    ("Pretraining makes the project work", "The same task went from 66.57% (small CNN from scratch) to about 95% by starting from ImageNet weights."),
    ("A bigger model is not automatically better", "ResNet50V2 has about 9 times the parameters of MobileNetV2 for 1.6 more points. EfficientNetB0's gain is within noise (p = 0.59)."),
    ("Explanations deserve little trust", "Grad-CAM did slightly worse than random pixel removal (0.361 vs 0.323), and different backbones highlighted different regions for the same correct prediction."),
    ("The smallest model is not the best to ship", "INT8 MobileNetV2 lost 17 points. The only compressed model that kept its accuracy was ResNet50V2, at 24.83 MB."),
    ("Distillation and pruning gave no clear benefit", "The student was neither much smaller nor faster. The pruning study used a weak, different network that also got extra training."),
    ("High confidence is not proof of being right", "Softmax saturates and the networks are overconfident. A black image was still labelled 'no tumor'."),
]

LIMITATIONS = [
    ("One split, one seed", "No cross-validation. The 95% interval on test accuracy is about plus or minus 1 to 1.3 points, and runs differ by about a point because training is not seeded."),
    ("Possible duplicates across the split", "Training and Testing folders were pooled and re-split. The dataset has 203 augmented meningioma images (Tr-aug-me, Te-aug-me), so copies can sit on both sides of the split and flatter the results."),
    ("Scan-type shortcut", "'No tumor' slices are mostly axial T2 while tumor slices are mostly T1 contrast, so the model may partly learn the scan type. Some slices carry stock-photo watermarks."),
    ("Resolution and augmentation", "128-pixel input and no augmentation in training."),
    ("One dataset", "No external test set, so we have not shown that the model works on images from other hospitals or scanners."),
    ("Compression experiments", "INT8 used only 100 calibration images and no quantization-aware training. There is no student trained without distillation. Pruning was run on a small separate CNN, not on the teacher."),
    ("Hardware", "Latency was measured on a Colab CPU, not on a phone or edge device."),
    ("Not a clinical tool", "This is a university research prototype. It is not a medical device and must not be used for diagnosis."),
]

REFERENCES = [
    "M. Sandler et al., MobileNetV2: Inverted residuals and linear bottlenecks, CVPR 2018.",
    "M. Tan and Q. V. Le, EfficientNet: Rethinking model scaling for convolutional neural networks, ICML 2019.",
    "K. He et al., Identity mappings in deep residual networks, ECCV 2016.",
    "R. R. Selvaraju et al., Grad-CAM: Visual explanations from deep networks via gradient-based localization, ICCV 2017.",
    "S. M. Lundberg and S.-I. Lee, A unified approach to interpreting model predictions, NeurIPS 2017.",
    "M. T. Ribeiro et al., 'Why should I trust you?': Explaining the predictions of any classifier, KDD 2016.",
    "G. Hinton, O. Vinyals and J. Dean, Distilling the knowledge in a neural network, arXiv:1503.02531, 2015.",
    "B. Jacob et al., Quantization and training of neural networks for efficient integer-arithmetic-only inference, CVPR 2018.",
    "Q. McNemar, Note on the sampling error of the difference between correlated proportions or percentages, Psychometrika 1947.",
]
