"""
A user-friendly web dashboard that uses our trained AI to automatically clean uploaded foggy images. It compares the raw image with the AI-enhanced version using YOLO object detection, ensuring the final result is always safer and more accurate through built-in enterprise safety guardrails.
"""

import os
import streamlit as st
import cv2
import numpy as np
import scipy.stats
import torch
from PIL import Image
from ultralytics import YOLO
from stable_baselines3 import PPO

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

CONFIDENCE_THRESHOLD = 0.40

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_PATH = os.path.join(BASE_DIR, "model", "clearsight_agent_ots_oracle.zip")
YOLO_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

@st.cache_resource
def load_models():
    yolo_model_path = YOLO_PATH if os.path.exists(YOLO_PATH) else 'yolov8n.pt'
    yolo = YOLO(yolo_model_path)
    yolo.to(device)
    agent = PPO.load(AGENT_PATH, device=device)
    return yolo, agent

def extract_state(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_bright = np.mean(gray)
    contrast = np.std(gray)
    hist = np.histogram(gray, bins=256, range=(0, 256))[0]
    hist = hist / (hist.sum() + 1e-5)
    entropy = scipy.stats.entropy(hist + 1e-5)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return np.array([mean_bright, contrast, entropy, blur], dtype=np.float32)

def apply_action(img, action_idx):
    if action_idx == 1: return cv2.convertScaleAbs(img, alpha=1.1, beta=15)
    elif action_idx == 2: return cv2.convertScaleAbs(img, alpha=0.9, beta=-15)
    elif action_idx == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    elif action_idx == 4:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    elif action_idx == 5:
        return cv2.bilateralFilter(img, 9, 75, 75)
    return img

def get_yolo_metrics(model, img):
    results = model(img, verbose=False, device=device)[0]
    valid_boxes = [box for box in results.boxes if box.conf[0].item() >= CONFIDENCE_THRESHOLD]
    total_conf = sum([box.conf[0].item() for box in valid_boxes])
    total_objects = len(valid_boxes)
    return total_conf, total_objects

def annotate_image(model, img):
    results = model(img, verbose=False, device=device)[0]
    annotated = img.copy()
    for box in results.boxes:
        conf = box.conf[0].item()
        if conf >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return annotated

st.set_page_config(page_title="ClearSight-RL Dashboard", layout="wide")
st.title("ClearSight-RL: Autonomous Vision Enhancer")
st.markdown(f"**Hardware Accelerated via:** `{device.upper()}` | **Safety Threshold:** `{CONFIDENCE_THRESHOLD * 100}%`")

try:
    yolo_model, rl_agent = load_models()
except Exception as e:
    st.error(f"Error loading models. Please ensure the model is located at: {AGENT_PATH}")
    st.stop()

uploaded_file = st.file_uploader("Upload a degraded or foggy image (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)
    
    with st.spinner("AI is analyzing and sequencing filters..."):
        current_img = raw_img.copy()
        used_actions = set()
        
        for step in range(6):
            state = extract_state(current_img)
            action, _ = rl_agent.predict(state, deterministic=True)
            act = int(action)
            
            if act in used_actions and act != 0:
                act = 0
            used_actions.add(act)
            
            if act == 0:
                break
            current_img = apply_action(current_img, act)
            
        raw_conf, raw_objs = get_yolo_metrics(yolo_model, raw_img)
        rl_conf, rl_objs = get_yolo_metrics(yolo_model, current_img)
        
        rollback_triggered = False
        if rl_conf < raw_conf:
            current_img = raw_img.copy()
            rl_objs = raw_objs
            rl_conf = raw_conf
            rollback_triggered = True
            
        raw_annotated = annotate_image(yolo_model, raw_img)
        rl_annotated = annotate_image(yolo_model, current_img)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Raw Camera Feed")
            st.image(cv2.cvtColor(raw_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.metric("Objects Detected", raw_objs)
            
        with col2:
            st.subheader("AI Enhanced Vision")
            st.image(cv2.cvtColor(rl_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.metric("Objects Detected", rl_objs, delta=int(rl_objs - raw_objs))
            
        if rollback_triggered:
            st.warning("**Safety Guardrail Triggered:** The AI enhancement was rejected because it lowered YOLO confidence. The system automatically rolled back to the raw image.")
        else:
            st.success("**Enhancement Successful:** The image was dynamically processed and passed all safety checks.")