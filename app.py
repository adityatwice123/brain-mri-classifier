
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np, cv2, streamlit as st, tensorflow as tf
import matplotlib.pyplot as plt

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
IMG_SIZE = 128

st.set_page_config(page_title='Brain MRI Classifier', page_icon='🧠', layout='wide')

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('teacher_model.keras', compile=False)

@st.cache_resource
def load_tflite():
    it = tf.lite.Interpreter(model_path='student_int8.tflite')
    it.allocate_tensors()
    return it

def get_gradcam(model, x):
    last = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)][-1].name
    gm = tf.keras.models.Model(model.inputs, [model.get_layer(last).output, model.output])
    with tf.GradientTape() as tape:
        conv, preds = gm(x[np.newaxis])
        score = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(score, conv)[0]
    w = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.maximum(tf.squeeze(conv[0] @ w[..., tf.newaxis]).numpy(), 0)
    cam = cv2.resize(cam, (224, 224))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

st.title('Brain Tumor MRI Classifier')
st.markdown('**Model:** MobileNetV2 | **Test accuracy:** 96.85% | **Dataset:** Brain Tumor MRI (4 classes)')
st.caption('Research prototype from a university project. Not a medical device and not for diagnosis.')
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader('Upload MRI Image')
    uploaded = st.file_uploader('Choose a brain MRI scan', type=['jpg', 'jpeg', 'png'])
    choice = st.radio('Select Model', ['Teacher (96.85% test acc, full precision)',
                                       'Student INT8 (70.37% test acc, 2.79 MB)'])

if uploaded:
    bgr = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        st.error('Could not read this file as an image.'); st.stop()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    shown = cv2.resize(rgb, (224, 224))
    x = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    with col1:
        st.image(shown, caption='Uploaded MRI Scan', use_container_width=True)

    teacher = load_model()
    if 'Teacher' in choice:
        probs = teacher.predict(x[np.newaxis], verbose=0)[0]
    else:
        it = load_tflite()
        it.set_tensor(it.get_input_details()[0]['index'], x[np.newaxis])
        it.invoke()
        probs = it.get_tensor(it.get_output_details()[0]['index'])[0]

    k = int(np.argmax(probs))
    pct = float(probs[k]) * 100
    with col2:
        st.subheader('Prediction Result')
        st.markdown(f'<h2 style="color:{COLORS[k]}">{CLASS_NAMES[k]}</h2>', unsafe_allow_html=True)
        st.metric('Model confidence', '99.9%+' if pct >= 99.9 else f'{pct:.1f}%')
        st.caption('Confidence is not the same as accuracy.')
        st.divider()
        st.write('**All class probabilities:**')
        for name, p in zip(CLASS_NAMES, probs):
            st.progress(float(p), text=f'{name}: {p*100:.1f}%')

    with col3:
        st.subheader('Grad-CAM Explanation')
        heat = plt.cm.jet(get_gradcam(teacher, x))[:, :, :3]
        over = np.clip(shown.astype(np.float32) / 255.0 * 0.6 + heat * 0.4, 0, 1)
        st.image((over * 255).astype(np.uint8), caption='Grad-CAM: red = where the model looked', use_container_width=True)
        st.caption('The heatmap always comes from the teacher model, and it is only a rough guide.')

    st.divider()
    a, b, c = st.columns(3)
    a.metric('Teacher test accuracy', '96.85%')
    b.metric('Student INT8 test accuracy', '70.37%')
    c.metric('Student INT8 file size', '2.79 MB')
    st.caption('Accuracy measured on 1,080 held-out test images from one dataset.')
