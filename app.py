import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

from utils.pose_utils import detect_pose_landmarks, compute_shoulder_angle, annotate_image_pil

# -------------------------
# Page Layout & Theme
# -------------------------
st.set_page_config(
    page_title="Frozen Shoulder Angle",
    layout="wide",   # <-- Wider layout
)

# Gmail-like light theme styling
st.markdown("""
<style>

/* ---------------------------------------------------
   GLOBAL LAYOUT — CENTER EVERYTHING
-----------------------------------------------------*/

html, body, .stApp {
    background-color: #f6f8fc !important;
    text-align: center !important;
}

/* Extra wide side padding */
.block-container {
    max-width: 1200px !important;     /* Wider container */
    padding-left: 8vw !important;
    padding-right: 8vw !important;
    padding-top: 2rem !important;
}

/* Force center alignment inside all containers */
div[data-testid="column"] > div {
    display: flex;
    flex-direction: column;
    align-items: center !important;
    justify-content: center !important;
    width: 100%;
}

/* ---------------------------------------------------
   MUCH BIGGER FONTS
-----------------------------------------------------*/

h1 {
    font-size: 3.2rem !important;
    font-weight: 700 !important;
    color: #202124 !important;
}

h2, h3, label, p, .stMarkdown {
    font-size: 1.4rem !important;
}

.stAlert {
    font-size: 1.3rem !important;
}

/* Radio buttons */
.stRadio label {
    font-size: 1.35rem !important;
}

/* Buttons */
button, .stDownloadButton button {
    font-size: 1.25rem !important;
    padding: 0.8rem 1.4rem !important;
    border-radius: 12px !important;
}

/* ---------------------------------------------------
   LARGE IMAGE DISPLAY
-----------------------------------------------------*/

img {
    width: 90% !important;          /* Make images bigger */
    max-width: 600px !important;    /* Prevent too large */
    border-radius: 12px;
    margin: 1rem auto !important;
    display: block;
}

/* Hover zoom for image */
img:hover {
    transform: scale(1.03);
    transition: 0.25s ease;
}

/* ---------------------------------------------------
   FILE UPLOADER + RADIO + HOVER EFFECTS
-----------------------------------------------------*/

.stFileUploader:hover {
    background-color: #ffffff !important;
    border: 2px solid #c6dafc !important;
    border-radius: 12px !important;
    transition: 0.2s ease-in-out;
}

.stRadio > label:hover, .st-radio label:hover {
    color: #1a73e8 !important;
    cursor: pointer;
    transition: 0.2s;
}

button:hover {
    transform: scale(1.05);
    background-color: #e8f0fe !important;
    transition: 0.15s;
}

/* ---------------------------------------------------
   RIGHT COLUMN = CARD DESIGN
-----------------------------------------------------*/

[data-testid="column"]:last-child {
    background: white;
    padding: 2rem !important;
    margin-top: 2rem !important;
    border-radius: 18px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    width: 100%;
}

/* ---------------------------------------------------
   MOBILE FIXES
-----------------------------------------------------*/
@media (max-width: 768px) {

    .block-container {
        padding-left: 4vw !important;
        padding-right: 4vw !important;
    }

    img {
        width: 100% !important;
        max-width: 360px !important;
    }

    button {
        width: 90% !important;
    }
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.title(" Frozen Shoulder Angle Detector")
st.markdown("""
Upload a **frontal image** of the patient showing the **shoulder and torso** clearly.
""")

# -------------------------
# Session State
# -------------------------
if "annotated_bytes" not in st.session_state:
    st.session_state.annotated_bytes = None
if "last_image" not in st.session_state:
    st.session_state.last_image = None
if "last_angle" not in st.session_state:
    st.session_state.last_angle = None
if "selected_side" not in st.session_state:
    st.session_state.selected_side = "left"

# -------------------------
# Layout (wider)
# -------------------------
col1, col2 = st.columns([3, 1])     # <-- wider left section

with col1:
    uploaded_file = st.file_uploader(" Choose an image", type=["jpg", "jpeg", "png"])

    side = st.radio(
        "Shoulder side to analyze",
        options=["left", "right"],
        index=0,
        horizontal=True
    )

    st.session_state.selected_side = side

    if uploaded_file is not None:
        # read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Unable to read the image. Try another file.")
        else:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = detect_pose_landmarks(image_rgb, min_detection_confidence=0.4)
            h, w = image_bgr.shape[:2]

            angle, landmark_positions = compute_shoulder_angle(results, w, h, side=side)

            annotated_pil = annotate_image_pil(
                image_bgr, results, angle, landmark_positions, side=side
            )

            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            annotated_bytes = buf.getvalue()

            st.session_state.annotated_bytes = annotated_bytes
            st.session_state.last_image = uploaded_file
            st.session_state.last_angle = angle

            st.image(
                annotated_pil,
                caption=f" Annotated image ({side} shoulder)",
                use_column_width=True
            )

            if angle is None:
                st.warning(
                    "Could not calculate the angle reliably. "
                    "Upload a clearer frontal image showing shoulder, elbow, and torso."
                )
            else:
                st.success(f"Estimated {side} shoulder angle: **{angle:.1f}°**")

with col2:
    st.markdown("###  Controls")
    if st.session_state.annotated_bytes:
        st.download_button(
            label="⬇ Download annotated image",
            data=st.session_state.annotated_bytes,
            file_name="annotated_shoulder.png",
            mime="image/png"
        )

# -------------------------
# Footer info
# -------------------------
st.markdown("---")
st.info(" **Tip:** Upload a clear frontal image with torso visible. Ensure elbow and hip are visible for best accuracy.")

# if st.session_state.last_angle is not None:
#     st.markdown(
#         f"### 📐 Last computed angle ({st.session_state.selected_side}): **{st.session_state.last_angle:.1f}°**"
#     )
