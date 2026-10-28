import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # must be set before TensorFlow is imported

import threading
from pathlib import Path

import altair as alt
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import content as C

ROOT = Path(__file__).parent
ASSETS, MODELS, SAMPLES = ROOT / "assets", ROOT / "models", ROOT / "samples"

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1280px;}
.hero {padding: 26px 30px; border-radius: 16px; margin-bottom: 18px; color: #fff;
       background: linear-gradient(120deg, #8E1B20 0%, #B3261E 55%, #D4552F 100%);}
.hero h1 {margin: 0 0 6px 0; font-size: 2.05rem; color: #fff; padding: 0;}
.hero p {margin: 0; font-size: 1.02rem; color: #fff; opacity: .95;}
.card {border: 1px solid rgba(128,128,128,.30); border-radius: 14px; padding: 14px 16px;
       background: rgba(128,128,128,.07); height: 100%;}
.card .k {font-size: .76rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em;}
.card .v {font-size: 1.75rem; font-weight: 700; line-height: 1.25;}
.card .s {font-size: .84rem; opacity: .75;}
.note {border-left: 4px solid #C26A00; padding: 9px 13px; background: rgba(194,106,0,.09);
       border-radius: 6px; font-size: .92rem; margin: 6px 0 12px 0;}
.good {border-left: 4px solid #2E8B57; padding: 9px 13px; background: rgba(46,139,87,.10);
       border-radius: 6px; font-size: .92rem; margin: 6px 0 12px 0;}
.bad {border-left: 4px solid #B3261E; padding: 9px 13px; background: rgba(179,38,30,.09);
      border-radius: 6px; font-size: .92rem; margin: 6px 0 12px 0;}
.pred {font-size: 2.1rem; font-weight: 800; line-height: 1.15; margin: 2px 0 2px 0;}
.small {font-size: .85rem; opacity: .75;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def _stretch(fn, *args, **kwargs):
    """Call a Streamlit element with full-width sizing, working across Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def show_img(src, **kw):
    return _stretch(st.image, src, **kw)


def show_df(df, **kw):
    return _stretch(st.dataframe, df, hide_index=True, **kw)


def show_chart(chart):
    return _stretch(st.altair_chart, chart)


def card(k, v, s=""):
    return f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>'


def cards(items):
    cols = st.columns(len(items))
    for col, (k, v, s) in zip(cols, items):
        col.markdown(card(k, v, s), unsafe_allow_html=True)


def note(text, kind="note"):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def figure(name, caption=None):
    show_img(str(ASSETS / name), caption=caption)


def wilson(acc_pct, n=C.N_TEST, z=1.96):
    p = acc_pct / 100
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half) * 100, (centre + half) * 100


def model_df():
    rows = []
    for m in C.MODELS:
        lo, hi = wilson(m["acc"])
        rows.append({
            "Model": m["name"], "Parameters (M)": m["params"], "Test accuracy (%)": m["acc"],
            "95% range (%)": f"{lo:.1f} to {hi:.1f}", "Macro F1": m["f1"],
            "INT8 accuracy (%)": m["int8_acc"], "Float32 weights (MB)": m["f32_mb"], "INT8 file (MB)": m["int8_mb"],
            "CPU latency (ms)": m["latency"],
        })
    return pd.DataFrame(rows)


def metrics_from_cm(cm):
    cm = np.array(cm, dtype=float)
    tp = np.diag(cm)
    prec = tp / cm.sum(0)
    rec = tp / cm.sum(1)
    f1 = 2 * prec * rec / (prec + rec)
    return pd.DataFrame({"Class": C.CLASS_NAMES, "Precision": prec.round(3), "Recall": rec.round(3), "F1": f1.round(3),
                         "Images": cm.sum(1).astype(int)})


def confusion_fig(cm, normalize=False):
    cm = np.array(cm, dtype=float)
    data = cm / cm.sum(1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(range(4)); ax.set_xticklabels(C.CLASS_NAMES, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(C.CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thr = data.max() * 0.55
    for i in range(4):
        for j in range(4):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(data[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white" if data[i, j] > thr else "#222")
    fig.tight_layout()
    return fig


def conf_text(pct):
    return "99.9%+" if pct >= 99.9 else f"{pct:.1f}%"


def prob_chart(probs):
    df = pd.DataFrame({"Class": C.CLASS_NAMES, "Probability (%)": np.round(np.asarray(probs) * 100, 1)})
    chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=4)
             .encode(x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
                     y=alt.Y("Class:N", sort=None, title=None),
                     color=alt.Color("Class:N", scale=alt.Scale(domain=C.CLASS_NAMES, range=C.CLASS_COLORS), legend=None),
                     tooltip=["Class", "Probability (%)"])
             .properties(height=150))
    return chart


# ----------------------------------------------------------------------------- pages
def page_overview():
    st.markdown('<div class="hero"><h1>Brain Tumor MRI Classification</h1>'
                '<p>Transfer learning, explainability and model compression on a four-class MRI dataset. '
                'Deep Learning Lab project, Symbiosis Institute of Technology, Pune.</p></div>', unsafe_allow_html=True)
    cards([("Dataset", "7,200", "MRI slices, 4 balanced classes"),
           ("Best single model", "96.85%", "ResNet50V2 test accuracy"),
           ("Reference model", "95.28%", "MobileNetV2 (teacher)"),
           ("Three-model vote", "97.50%", "soft-voting ensemble")])
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("What we did")
        figure("pipeline.png", "Project pipeline")
    with right:
        st.subheader("How to use this app")
        st.markdown(
            "- **Dataset**: counts, split, samples and preprocessing\n"
            "- **Architecture & training**: how each model is built and trained\n"
            "- **Results**: compare models, explore confusion matrices\n"
            "- **Explainable AI**: Grad-CAM, SHAP, LIME and the faithfulness test\n"
            "- **Compression**: distillation, pruning and INT8 quantization\n"
            "- **Live demo**: classify a slice and see the Grad-CAM overlay\n"
            "- **Testing & CI**, **Insights & limitations** and **About**")
        note("All numbers on these pages come from the documented notebook run. Training is not seeded, so a new run "
             "can differ by about one point.")
    st.subheader("Key findings")
    cols = st.columns(3)
    for i, (title, text) in enumerate(C.INSIGHTS):
        with cols[i % 3]:
            st.markdown(f"**{title}**")
            st.write(text)
    note("Research prototype. Not a medical device and not for diagnosis.", "bad")


def page_dataset():
    st.header("Dataset")
    t1, t2, t3 = st.tabs(["Counts and split", "Sample gallery", "Preprocessing"])
    with t1:
        df = pd.DataFrame(C.SPLIT, columns=["Class", "Total", "Train", "Validation", "Test"])
        total = df[["Total", "Train", "Validation", "Test"]].sum()
        c1, c2 = st.columns([2, 3])
        with c1:
            show_df(pd.concat([df, pd.DataFrame([["All classes", *total.tolist()]], columns=df.columns)]))
            st.markdown("- 70% / 15% / 15% stratified split with seed 42.\n"
                        "- Train learns, validation tunes and stops early, test is touched once at the end.\n"
                        "- Perfectly balanced, so plain accuracy is a fair metric.")
        with c2:
            long = df.melt(id_vars="Class", value_vars=["Train", "Validation", "Test"], var_name="Split", value_name="Images")
            chart = (alt.Chart(long).mark_bar().encode(
                x=alt.X("Class:N", sort=None, title=None), y=alt.Y("Images:Q"),
                color=alt.Color("Split:N", scale=alt.Scale(range=["#B3261E", "#C26A00", "#1F5FA8"])),
                xOffset="Split:N", tooltip=["Class", "Split", "Images"]).properties(height=300))
            show_chart(chart)
        with st.expander("Figure from the notebook (Step 7)"):
            figure("class_distribution.png")
        note("The dataset contains 203 augmented meningioma images (files named Tr-aug-me and Te-aug-me) that were added to "
             "balance the class. Because the original Training and Testing folders were pooled and re-split, copies can sit on "
             "both sides of the split. See Insights & limitations.")
    with t2:
        figure("sample_mri.png", "Five sample slices per class (rows: glioma, meningioma, no tumor, pituitary)")
        st.subheader("Browse the sample slices used in the live demo")
        cls = st.radio("Class folder", C.CLASS_NAMES, horizontal=True, key="gal_cls")
        code = [k for k, v in C.FOLDER_CODE.items() if v == C.CLASS_NAMES.index(cls)][0]
        files = sorted(SAMPLES.glob(f"Te-{code}_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
        cols = st.columns(len(files))
        for col, f in zip(cols, files):
            with col:
                show_img(str(f), caption=f.name)
        st.caption("These slices come from the dataset's original Testing folder. The notebook pooled and re-split all images, "
                   "so a sample may have been seen during training.")
    with t3:
        show_df(pd.DataFrame(C.PREPROCESS, columns=["Step", "What we did", "Why"]))
        figure("augmentation.png", "Augmentation examples (pipeline built, not used in training)")


def page_architecture():
    st.header("Architecture and training")
    t1, t2, t3 = st.tabs(["Shared design", "The models", "Training"])
    with t1:
        st.markdown("Every model is a **pretrained backbone plus one shared classification head**, so differences in results come "
                    "from the backbone and not from the head.")
        figure("architecture.png", "Block diagram of the classifier")
        show_df(pd.DataFrame(C.LAYERS, columns=["Layer", "What it does", "Why"]))
        with st.expander("What is transfer learning?"):
            st.write("Start from a network already trained on ImageNet. Its early layers detect edges and textures that also help "
                     "on MRI. With only 5,040 training images this works far better than training from scratch. In this project a "
                     "small CNN from scratch reached 66.57%, while pretrained MobileNetV2 reached 95.28%.")
        with st.expander("Softmax and confidence"):
            st.write("Softmax turns four scores into probabilities that sum to 1. The largest one is the model's confidence. "
                     "It says how strongly the model favours a class, not how likely it is to be right.")
    with t2:
        names = [m["name"] for m in C.MODELS]
        pick = st.selectbox("Choose a model", names, key="arch_model")
        m = next(x for x in C.MODELS if x["name"] == pick)
        info = C.MODEL_INFO[pick]
        cards([("Parameters", f'{m["params"]}M', m["params_exact"]), ("Test accuracy", f'{m["acc"]}%', f'macro F1 {m["f1"]}'),
               ("INT8 accuracy", f'{m["int8_acc"]}%', f'{m["int8_mb"]} MB file'), ("CPU latency", f'{m["latency"]} ms', "per image, INT8")])
        st.write("")
        st.markdown(f"**What it is.** {info['what']}")
        st.markdown("**How it works**")
        for h in info["how"]:
            st.markdown(f"- {h}")
        st.markdown(f"**Result in this project.** {info['result']}")
        st.markdown(f"**Why.** {info['why']}")
        st.divider()
        chart = (alt.Chart(model_df()).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Parameters (M):Q"), y=alt.Y("Model:N", sort=None, title=None),
            color=alt.Color("Model:N", legend=None, scale=alt.Scale(domain=names, range=[x["color"] for x in C.MODELS])),
            tooltip=["Model", "Parameters (M)"]).properties(height=170, title="Model size in parameters (millions)"))
        show_chart(chart)
    with t3:
        st.markdown("Training used two phases. The head is trained first on a frozen backbone, then the top layers are "
                    "unfrozen and trained gently.")
        show_df(pd.DataFrame(C.TRAINING, columns=["Setting", "Phase A", "Phase B"]))
        st.markdown("**Why it is done this way.** The head starts random, and its large early gradients would wreck the pretrained "
                    "features, so the backbone is frozen first. Phase B uses a 10 times lower learning rate so small steps adjust "
                    "weights without destroying them.")
        rows = [{"Model": m["name"], "Best val. accuracy, Phase A": f'{m["best_val_a"]:.4f}',
                 "Best val. accuracy, Phase B": f'{m["best_val_b"]:.4f}',
                 "Training time (min)": "not recorded" if m["train_min"] is None else f'{m["train_min"]}'} for m in C.MODELS[:3]]
        show_df(pd.DataFrame(rows))
        figure("teacher_curves.png", "MobileNetV2 training curves (dashed line: start of fine-tuning)")
        figure("three_model_curves.png", "Validation accuracy, validation loss and training accuracy of the three backbones")
        note("Training accuracy reached 99.7% while validation stayed near 94.4%, so the teacher overfits a little. For "
             "EfficientNetB0 and ResNet50V2 we kept BatchNorm layers frozen in Phase B, while the MobileNetV2 teacher's stayed "
             "trainable, so the comparison is not perfectly identical.")


def page_results():
    st.header("Results")
    t1, t2, t3 = st.tabs(["Compare models", "Confusion matrices", "Significance"])
    df = model_df()
    with t1:
        chosen = st.multiselect("Models to compare", [m["name"] for m in C.MODELS], default=[m["name"] for m in C.MODELS[:3]])
        sub = df[df["Model"].isin(chosen)]
        if sub.empty:
            st.info("Pick at least one model.")
        else:
            metric = st.radio("Metric", ["Test accuracy (%)", "Macro F1", "INT8 accuracy (%)"], horizontal=True)
            lo, hi = {"Test accuracy (%)": (90, 100), "Macro F1": (0.90, 1.0), "INT8 accuracy (%)": (60, 100)}[metric]
            chart = (alt.Chart(sub).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
                y=alt.Y(f"{metric}:Q", scale=alt.Scale(domain=[lo, hi], clamp=True)),
                y2=alt.Y2Datum(lo),
                color=alt.Color("Model:N", legend=None, scale=alt.Scale(domain=[m["name"] for m in C.MODELS], range=[m["color"] for m in C.MODELS])),
                tooltip=["Model", metric]).properties(height=320))
            labels = chart.mark_text(dy=-8, fontWeight="bold").encode(text=f"{metric}:Q")
            show_chart(chart + labels)
            st.caption(f"The axis starts at {lo}, not at zero, so that small differences are visible.")
            show_df(sub)
        st.caption(f"The 95% range is a binomial interval on {C.N_TEST} test images, so gaps under about one point are within noise.")
        cards([("Soft-vote ensemble", f'{C.ENSEMBLE["acc"]}%', f'macro F1 {C.ENSEMBLE["f1"]} (average of the three backbones)'),
               ("Small CNN from scratch", f"{C.SMALL_CNN_ACC}%", "baseline: shows why pretraining matters")])
        st.write("")
        per = pd.DataFrame([{"Model": n, **{c: round(metrics_from_cm(cm)["F1"][i], 3) for i, c in enumerate(C.CLASS_NAMES)}}
                            for n, cm in C.CONFUSION.items()])
        st.markdown("**Per-class F1**")
        show_df(per)
        figure("three_f1.png", "Per-class F1 of the three backbones")
    with t2:
        c1, c2 = st.columns([1, 1])
        with c1:
            model = st.selectbox("Model", list(C.CONFUSION), key="cm_model")
            norm = st.toggle("Show row-normalised values", value=False)
            cm = np.array(C.CONFUSION[model])
            st.pyplot(confusion_fig(cm, norm))
        with c2:
            st.markdown("**Precision, recall and F1**")
            show_df(metrics_from_cm(cm))
            off = cm.copy(); np.fill_diagonal(off, 0)
            i, j = np.unravel_index(off.argmax(), off.shape)
            note(f"Most common mistake: {C.CLASS_NAMES[i]} read as {C.CLASS_NAMES[j]} ({off[i, j]} of {cm[i].sum()} images). "
                 "Glioma versus meningioma is the main source of error for all three backbones.")
            st.caption(f"Accuracy: {np.trace(cm) / cm.sum() * 100:.2f}% ({np.trace(cm)} of {cm.sum()} test images correct).")
        with st.expander("Figures from the notebook"):
            figure("teacher_confusion.png", "Teacher confusion matrix")
            figure("three_confusion.png", "Confusion matrices of the three backbones")
    with t3:
        st.markdown("McNemar's test compares two models on the same images, using only the images where they disagree. "
                    "A p-value below 0.05 means a real difference.")
        rows = [{"Compared with MobileNetV2": n, "MobileNetV2 right, other wrong": a, "Other right, MobileNetV2 wrong": b,
                 "p-value": p, "Verdict": "real difference" if p < 0.05 else "not different"} for n, a, b, p in C.MCNEMAR]
        show_df(pd.DataFrame(rows))
        note("ResNet50V2 fixed 35 images the teacher got wrong and broke 18, a real gain (p = 0.028). EfficientNetB0 fixed 30 and "
             "broke 25, which is within noise (p = 0.59), so we cannot say it beats MobileNetV2.", "good")


def page_xai():
    st.header("Explainable AI")
    method = st.radio("Method", list(C.XAI), horizontal=True)
    x = C.XAI[method]
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader(f"{method}: how it works")
        for i, h in enumerate(x["how"], 1):
            st.markdown(f"{i}. {h}")
        st.subheader("What we found")
        st.write(x["found"])
        note(x["caveat"])
    with c2:
        for name, cap in x["figs"]:
            figure(name, cap)
    if method == "Deletion test":
        st.divider()
        cards([("Grad-CAM order", "0.361", "area under the confidence curve"), ("Random order", "0.323", "same number of pixels removed"),
               ("Verdict", "Not faithful", "lower area for Grad-CAM was needed")])
    st.caption("For a live Grad-CAM on any slice, open the Live demo page.")


def page_compression():
    st.header("Model compression")
    t1, t2, t3, t4, t5 = st.tabs(["Trade-off explorer", "Knowledge distillation", "Pruning", "INT8 quantization", "Benchmark"])
    df = model_df()
    with t1:
        st.markdown("Explore how accuracy trades against size and speed.")
        c1, c2 = st.columns(2)
        xaxis = c1.selectbox("X axis", ["INT8 file (MB)", "CPU latency (ms)", "Float32 weights (MB)", "Parameters (M)"])
        yaxis = c2.selectbox("Y axis", ["Test accuracy (%)", "INT8 accuracy (%)"])
        base = alt.Chart(df).encode(
            x=alt.X(f"{xaxis}:Q", scale=alt.Scale(zero=False, padding=30)), y=alt.Y(f"{yaxis}:Q", scale=alt.Scale(zero=False, padding=18)),
            color=alt.Color("Model:N", legend=None, scale=alt.Scale(domain=[m["name"] for m in C.MODELS], range=[m["color"] for m in C.MODELS])),
            tooltip=["Model", xaxis, yaxis])
        pts = base.mark_circle(size=260, opacity=.9)
        txt = base.mark_text(align="left", dx=14, dy=-4, fontSize=12).encode(text="Model:N")
        show_chart((pts + txt).properties(height=380))
        show_df(df)
        note("The notebook's Keras file sizes (26.18, 33.22, 217.06 and 9.89 MB) also hold optimizer state, so they overstate the "
             "reduction. The honest comparison is against float32 weights (parameters times 4 bytes): about 3.3 to 3.9 times smaller.")
    with t2:
        st.markdown("A small **student** learns from the big **teacher's** probabilities, which carry extra information "
                    "(for example 85% glioma, 12% meningioma).")
        st.latex(r"L = 0.7 \cdot T^2 \cdot KL(p_{teacher}^{T} \,\|\, p_{student}^{T}) + 0.3 \cdot CE(y, p_{student})")
        cards([("Temperature", "T = 4", "softens probabilities"), ("Student accuracy", "91.67%", "macro F1 0.9159"),
               ("Drop vs teacher", "3.61 pts", "95.28% to 91.67%"), ("Distillation loss", "7.84 to 0.34", "over 15 epochs")])
        st.write("")
        st.markdown("- The student is a MobileNetV2 with a 64-unit head and 814,724 trainable parameters (teacher: 1,890,308).\n"
                    "- Both networks end in softmax, so we took a logarithm to recover logits before softening them.\n"
                    "- The student is barely smaller (2.34M against 2.62M parameters) and not faster (about 4 ms each).\n"
                    "- No student was trained without distillation, so we cannot claim distillation helped.")
        figure("kd_curves.png", "Distillation training curves")
    with t3:
        pdf = pd.DataFrame(C.PRUNING, columns=["Target sparsity", "Measured sparsity", "Test accuracy (%)", "Weights after gzip (MB)"])
        c1, c2 = st.columns([2, 3])
        with c1:
            show_df(pdf)
        with c2:
            ch = pd.DataFrame({"Sparsity (%)": [0, 50, 70, 90], "Accuracy (%)": [r[2] for r in C.PRUNING]})
            line = alt.Chart(ch).mark_line(point=alt.OverlayMarkDef(size=90)).encode(
                x="Sparsity (%):Q", y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[0, 100])), tooltip=["Sparsity (%)", "Accuracy (%)"])
            show_chart(line.properties(height=260))
        note("This experiment used a small separate CNN, not the teacher. The 50% model beats the dense base by about 19 points, "
             "which pruning cannot explain: the pruned models got ten extra epochs of training on top of an under-trained base. "
             "The only clean result is that at 90% sparsity this network collapses to 50%.")
        figure("pruning_curve.png", "Accuracy of the small CNN at each pruning level")
    with t4:
        st.markdown("INT8 post-training quantization stores weights and activations as 8-bit integers, "
                    "using a scale and zero-point. We used TensorFlow Lite with 100 calibration images and float32 input and output.")
        q = pd.DataFrame([{"Model": m["name"], "Float accuracy (%)": m["acc"], "INT8 accuracy (%)": m["int8_acc"],
                           "Change (points)": round(m["int8_acc"] - m["acc"], 2), "INT8 macro F1": m["int8_f1"],
                           "INT8 file (MB)": m["int8_mb"], "Size cut vs float32": f'{m["f32_mb"] / m["int8_mb"]:.1f}x'} for m in C.MODELS])
        show_df(q)
        long = pd.DataFrame([{"Model": m["short"], "Version": v, "Accuracy (%)": a} for m in C.MODELS
                             for v, a in (("Float", m["acc"]), ("INT8", m["int8_acc"]))])
        ch = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[60, 100])),
            color=alt.Color("Version:N", scale=alt.Scale(range=["#9AA0A6", "#B3261E"])), xOffset="Version:N",
            tooltip=["Model", "Version", "Accuracy (%)"]).properties(height=300)
        show_chart(ch)
        note("ResNet50V2 came through quantization untouched. MobileNetV2 lost 17 points and its student nearly 22. Depthwise "
             "networks are known to quantize poorly without special handling, and we calibrated with only 100 images, but we did "
             "not test the cause. The smallest model was not safe to compress.", "bad")
        figure("compression_results.png", "Size and accuracy for the teacher, student and pruned networks")
        figure("gradcam_teacher_vs_student.png", "Grad-CAM of the teacher and the distilled student")
    with t5:
        figure("benchmark.png", "Four-model benchmark: file size, accuracy before and after INT8, accuracy against CPU latency")
        cards([("Most accurate after INT8", "ResNet50V2", "96.85%"), ("Smallest", "Student (KD)", "2.79 MB"), ("Fastest on CPU", "Student (KD)", "4.06 ms")])
        st.caption("Latency is the mean CPU time per image over 100 runs after 5 warm-up runs, single thread, on a Colab runtime.")


# ----------------------------------------------------------------------------- live demo
_LOCK = threading.Lock()


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_models():
    import tensorflow as tf
    teacher = tf.keras.models.load_model(str(MODELS / "teacher_model.keras"), compile=False)
    last = [l for l in teacher.layers if isinstance(l, tf.keras.layers.Conv2D)][-1].name
    grad_model = tf.keras.models.Model(teacher.inputs, [teacher.get_layer(last).output, teacher.output])
    interp = tf.lite.Interpreter(model_path=str(MODELS / "student_int8.tflite"))
    interp.allocate_tensors()
    return teacher, grad_model, interp


def preprocess(raw):
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    shown = cv2.resize(rgb, (224, 224))
    x = cv2.resize(rgb, (128, 128)).astype(np.float32) / 255.0   # same preprocessing as training
    return x, shown


def predict_teacher(x):
    teacher, _, _ = load_models()
    with _LOCK:
        return np.asarray(teacher(x[None], training=False))[0]


def predict_student(x):
    _, _, it = load_models()
    with _LOCK:
        it.set_tensor(it.get_input_details()[0]["index"], x[None].astype(np.float32))
        it.invoke()
        return it.get_tensor(it.get_output_details()[0]["index"])[0]


def gradcam(x):
    import tensorflow as tf
    _, gm, _ = load_models()
    with _LOCK:
        with tf.GradientTape() as tape:
            conv, preds = gm(x[None])
            score = preds[:, tf.argmax(preds[0])]
        grads = tape.gradient(score, conv)[0]
        w = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.maximum(tf.squeeze(conv[0] @ w[..., tf.newaxis]).numpy(), 0)
    cam = cv2.resize(cam, (224, 224))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def sample_true_class(name):
    try:
        return C.FOLDER_CODE[name.split("-")[1][:2]]
    except Exception:
        return None


def result_block(title, probs, true_idx):
    k = int(np.argmax(probs))
    pct = float(probs[k]) * 100
    st.markdown(f"**{title}**")
    st.markdown(f'<div class="pred" style="color:{C.CLASS_COLORS[k]}">{C.CLASS_NAMES[k]}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="small">Model confidence</span> **{conf_text(pct)}**', unsafe_allow_html=True)
    if true_idx is not None:
        if k == true_idx:
            note(f"Matches the folder label ({C.CLASS_NAMES[true_idx]}).", "good")
        else:
            note(f"Folder label is {C.CLASS_NAMES[true_idx]}, so this is a wrong prediction.", "bad")
    show_chart(prob_chart(probs))


def page_live():
    st.header("Live demo")
    note("Confidence is not accuracy. A softmax always picks one of the four classes, even for an image that is not a brain MRI. "
         "The accuracy figures in this app come from the notebook run (Results page). The model files here are exported "
         "checkpoints, and since training is not seeded their accuracy can differ by about a point. INT8 outputs also move in "
         "steps of about 0.4%, so a very confident INT8 answer shows as 99.6%.")
    ctrl, out = st.columns([1, 2.2], gap="large")
    raw, name, true_idx = None, None, None
    with ctrl:
        src = st.radio("Image source", ["Sample slices", "Upload an image"], horizontal=True)
        mode = st.radio("Model", ["Teacher (full precision)", "Student INT8", "Compare both"])
        alpha = st.slider("Grad-CAM overlay strength", 0.0, 1.0, 0.45, 0.05)
        if src == "Sample slices":
            cls = st.selectbox("Class folder", C.CLASS_NAMES)
            code = [k for k, v in C.FOLDER_CODE.items() if v == C.CLASS_NAMES.index(cls)][0]
            files = sorted(SAMPLES.glob(f"Te-{code}_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
            name = st.radio("Pick a slice", [f.name for f in files], horizontal=True)
            raw = (SAMPLES / name).read_bytes()
            true_idx = sample_true_class(name)
            st.caption("Samples come from the dataset's original Testing folder. The notebook pooled and re-split all images, "
                       "so a sample may have been seen in training.")
        else:
            up = st.file_uploader("Upload a brain MRI slice (jpg or png)", type=["jpg", "jpeg", "png"])
            if up is not None:
                raw, name = up.read(), up.name
            st.caption("The app cannot tell whether an upload is really a brain MRI.")
    with out:
        if raw is None:
            st.info("Choose a sample slice or upload an image to see the prediction.")
            return
        x, shown = preprocess(raw)
        if x is None:
            st.error("Could not read this file as an image.")
            return
        with st.spinner("Running the model..."):
            use_t = mode in ("Teacher (full precision)", "Compare both")
            use_s = mode in ("Student INT8", "Compare both")
            p_t = predict_teacher(x) if use_t else None
            p_s = predict_student(x) if use_s else None
            cam = gradcam(x)
        heat = plt.get_cmap("jet")(cam)[:, :, :3]
        over = np.clip(shown.astype(np.float32) / 255.0 * (1 - alpha) + heat * alpha, 0, 1)
        top = st.columns(2)
        with top[0]:
            show_img(shown, caption=f"Input ({name}), resized to 224 x 224 for display")
        with top[1]:
            show_img((over * 255).astype(np.uint8), caption="Grad-CAM overlay (always from the teacher): red = where the model looked")
        st.divider()
        blocks = []
        if use_t:
            blocks.append(("Teacher (MobileNetV2, full precision)", p_t))
        if use_s:
            blocks.append(("Student INT8 (distilled MobileNetV2)", p_s))
        cols = st.columns(len(blocks))
        for col, (title, probs) in zip(cols, blocks):
            with col:
                result_block(title, probs, true_idx)
        if use_t and use_s:
            a, b = int(np.argmax(p_t)), int(np.argmax(p_s))
            note("Both models agree." if a == b else f"The models disagree: teacher says {C.CLASS_NAMES[a]}, student says {C.CLASS_NAMES[b]}.",
                 "good" if a == b else "note")
        st.caption("Reported test accuracy (notebook run): teacher 95.28%, student INT8 69.91%, measured on 1,080 held-out images. "
                   "One image says nothing about accuracy.")


def page_testing():
    st.header("Testing and CI")
    cards([("pytest", "12 passed", "0 failed (run in Colab, 11.92 s)"), ("Data tests", "7", "class count, split, shape, scaling, NaN"),
           ("Model tests", "5", "output shape, probabilities, TFLite, size, accuracy")])
    st.write("")
    t = pd.DataFrame(C.PYTEST, columns=["File", "Test", "What it checks"])
    t.insert(3, "Result", "PASSED")
    show_df(t)
    st.subheader("GitHub Actions workflow")
    st.markdown("The workflow below runs the same tests on every push and pull request.")
    figure("ci_diagram.png", "CI pipeline defined in .github/workflows/ci.yml")
    st.code(C.CI_YAML, language="yaml")
    note("The pytest run happened in the Colab notebook. The workflow is the CI design for the GitHub repository.")


def page_insights():
    st.header("Insights and limitations")
    st.subheader("Key insights")
    for i, (title, text) in enumerate(C.INSIGHTS, 1):
        st.markdown(f"**{i}. {title}.** {text}")
    st.subheader("Limitations")
    for title, text in C.LIMITATIONS:
        with st.expander(title):
            st.write(text)
    st.subheader("Future work")
    st.markdown("- Augmentation and 224-pixel inputs.\n- k-fold evaluation on a deduplicated split.\n"
                "- Quantization-aware training for MobileNetV2.\n- A student trained without distillation, and pruning on the real teacher.\n"
                "- Explanation methods scored against tumor masks, and an external test set from another hospital.")


def page_about():
    st.header("About")
    c1, c2 = st.columns([1, 2])
    with c1:
        show_img(str(ASSETS / "sit_logo.png"))
    with c2:
        st.markdown("**Symbiosis Institute of Technology, Pune**  \nDepartment of Artificial Intelligence and Machine Learning  \n"
                    "Deep Learning Lab, Semester 5, Academic Year 2026-27")
        st.markdown(f"**Under the guidance of** {C.GUIDE}")
    show_df(pd.DataFrame(C.TEAM, columns=["Team member", "PRN"]))
    st.subheader("Data")
    st.write("A public four-class brain tumor MRI dataset (glioma, meningioma, no tumor, pituitary), 7,200 images, "
             "pooled from its Training and Testing folders and re-split 70/15/15.")
    st.subheader("Methods referenced")
    for r in C.REFERENCES:
        st.markdown(f"- {r}")
    note("Research prototype for a university project. Not a medical device and not for diagnosis.", "bad")


# ----------------------------------------------------------------------------- navigation
PAGES = {
    "Overview": page_overview, "Dataset": page_dataset, "Architecture & training": page_architecture,
    "Results": page_results, "Explainable AI": page_xai, "Compression": page_compression,
    "Live demo": page_live, "Testing & CI": page_testing, "Insights & limitations": page_insights, "About": page_about,
}

with st.sidebar:
    st.image(str(ASSETS / "sit_logo.png"), width=90)
    st.markdown("### Brain Tumor MRI")
    st.caption("Classification, explainability and compression")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Research prototype. Not for diagnosis.")

PAGES[choice]()
