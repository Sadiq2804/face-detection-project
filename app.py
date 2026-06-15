import streamlit as st
from PIL import Image
import tempfile, os, io
from detector import detect_faces

st.set_page_config(page_title="Face Detector", page_icon="👤")
st.title("👤 Face Detection App")
st.caption("Powered by MediaPipe + OpenCV")

confidence = st.slider("Min confidence", 0.1, 1.0, 0.5, 0.05)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(uploaded, use_column_width=True)
    with col2:
        st.subheader("Detected")
        result = detect_faces(tmp_path, confidence)
        st.image(result, use_column_width=True)

    os.unlink(tmp_path)
    result_img = Image.fromarray(result)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    st.download_button("⬇️ Download result", buf.getvalue(),
                       file_name="faces_detected.jpg", mime="image/jpeg")
