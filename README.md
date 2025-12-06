# Frozen Shoulder Angle Detector

Web app that uses MediaPipe Pose to estimate left/right shoulder angles from an uploaded image. Built with Streamlit.

## Features
- Upload image (jpg, png)
- Choose left or right shoulder
- Calculates shoulder angle (shoulder-elbow vs shoulder-hip)
- Annotated image shown and downloadable
- Clear/reset button
- Deployable on Streamlit Cloud

## Folder structure
See repository root.

## How to run locally
1. Create and activate a Python virtual environment.
2. Install dependencies:

pip install -r requirements.txt
3. Run app:

streamlit run app.py

## Deployment on Streamlit Cloud
1. Create a GitHub repository and push the project (all files).
2. Go to https://cloud.streamlit.com/ and sign in with GitHub.
3. Click "New app" → Choose repository, branch, and `app.py` as main file.
4. Deploy. Streamlit Cloud will install packages from `requirements.txt`.
5. Share the generated link with doctors.

## Notes
- Best results when the image contains the torso, shoulder, elbow, and hip.
- Angle definition: angle between upper-arm vector and torso vector at the shoulder.
- You may adjust MediaPipe detection confidence in `utils/pose_utils.py`.
