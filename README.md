# **ClearSight-RL: Autonomous Image Enhancement using Reinforcement Learning**

ClearSight-RL is a Reinforcement Learning pipeline that dynamically enhances degraded images (e.g., heavy fog) to maximize YOLOv8 object detection accuracy. It trains a PPO agent to intelligently sequence OpenCV filters and uses inference-time guardrails to guarantee the enhancement never degrades baseline performance.

![hero_highlight_92](https://github.com/user-attachments/assets/4068dff6-1a5f-44f7-a9e2-260df448c3c5)

*Comparison of standard YOLOv8 detection on a foggy image (left) vs. ClearSight-RL enhanced detection (right).*

## **Project Structure**

ClearSight-RL/  
├── README.md  
├── inference.py  
├── requirements.txt  
├── yolov8n.pt  
└── model/  
    └── clearsight\_agent\_ots\_oracle.zip

## **1\. Model Training (Kaggle)**

Due to the size of the datasets (RTTS, SOTS, Foggy Cityscapes) and the compute resources required, the primary training pipeline is designed to be run in cloud environments like Kaggle.

* **Training Notebook:** https://www.kaggle.com/code/kkm121121/train-clearsight/notebook

**Training Details:**

* **Algorithm:** Proximal Policy Optimization (PPO) via stable-baselines3  
* **Reward Signal:** Change in YOLOv8 Confidence and Bounding Box Intersection over Union (IoU)  
* **Optimization:** Subprocess Vectorization for parallel environment rollout

## **2\. Local Inference Dashboard**

A local Streamlit dashboard is provided to test the trained agent on custom images and visualize the filter sequencing in real-time.

### **Prerequisites**

Ensure you have Python 3.9+ installed. Clone this repository and install the dependencies:

pip install \-r requirements.txt

### **Running the Dashboard**

1. Ensure the pre-trained agent (clearsight\_agent\_ots\_oracle.zip) is located in the model/ directory.  
2. Launch the Streamlit application:

streamlit run inference.py

## **3\. Benchmarking & Guardrails**

The project includes benchmarking scripts to evaluate the agent against standard datasets (e.g., Foggy Cityscapes). To accurately simulate real-world constraints (like autonomous driving safety thresholds), the benchmark filters out object detections with low confidence scores (e.g., \< 0.60).

### **Inference Guardrails**

To prevent the agent from over-processing images, two primary constraints are applied during inference:

* **Action Masking:** Prevents the RL agent from recursively applying the same high-contrast filter, avoiding image degradation (noise/artifacts).  
* **Confidence Rollbacks:** Compares the YOLO detection confidence of the raw image against the AI-enhanced image. If the enhancement reduces the overall detection confidence (e.g., due to domain shift), the system reverts to the raw image. This ensures the pipeline's output is strictly greater than or equal to the baseline performance.

### **Results (Foggy Cityscapes)**

Evaluated on 500 images with a strict 0.60 confidence threshold:

* **Raw Foggy Detections:** 1,126 objects  
* **RL Enhanced Detections:** 1,228 objects  
* **Performance Delta:** \+9.1% increase in high-confidence object detections
