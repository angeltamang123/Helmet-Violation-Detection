# Helmet Violation Detector

This application is an intellignet system designed to identify motorcycle helmet violations from traffic camera footage. It utilizes deep learning models (YOLO11 and YOLO26) and motion-tracking heuristics to ensure fair and accurate reporting for traffic enforcement.

---

## 1. System Requirements & Libraries
The user needs to download the zip file for the intelligent system. This folder should at least contain four files: model files for the two models, the source code and the text file to download required libraries with few python commands.
 
The following libraries are required to run the application. These handle neural network inference, image processing, and the dashboard interface.

* **ultralytics**: The core engine for YOLO11/YOLO26 object detection and tracking.
* **streamlit**: Provides the interactive web interface and forensic dashboard.
* **opencv-python-headless**: Used for video frame manipulation and drawing evidence markers.
* **numpy**: Handles mathematical calculations for the **Stationary Motion Filter**.
* **pillow**: Manages image formatting for the violation evidence gallery.

---

## 2. Environment Setup
It is highly recommended to run this software in a virtual environment to avoid library conflicts and on stable older python versions such as 3.12.13.

### Option A: Using Standard Python (venv)
1. Open your terminal in the project folder.
2. Create the environment:  
   `python -m venv helmet_env`
3. Activate it:
   - **Windows**: `.\helmet_env\Scripts\activate`
   - **Mac/Linux**: `source helmet_env/bin/activate`
4. Install libraries:  
   `pip install -r requirements.txt`

### Option B: Using Conda (Recommended)
1. Open your Anaconda Prompt.
2. Create the environment:  
   `conda create --name helmet_env python=3.12.13 -y`
3. Activate it:  
   `conda activate helmet_env`
4. Install libraries:  
   `pip install -r requirements.txt`

---

## 3. Input Data Specifications
To ensure high accuracy in detection and tracking, input data must meet these criteria:

* **Format**: `.mp4`, `.avi`, or `.mov` video files.
* **File Size**: Maximum **50MB** per upload.
* **Resolution**: 720p or higher is recommended to clearly distinguish helmet straps and textures.

---

## 4. Software Operation Guide

### Step 1: Selection & Configuration
In the sidebar, select the Model Engine

### Step 2: Analysis Pipeline
Upload the video file. The system will automatically:
1. **Process Frames**: The system analyzes every 3rd frame to balance speed and accuracy.
2. **Filter Stationary Objects**: Using Euclidean distance, the system ignores riders who are parked or stationary (movement threshold < 3 pixels).
3. **Evidence Capture**: For every unique ID detected, the system saves only the "Best Frame" (the frame with the highest detection confidence).

### Step 3: Review
Once "Analysis Complete" appears, use the navigation bar:
* **Arrows**: Use `Previous` and `Next` to cycle through unique violators. The buttons automatically disable when you reach the start or end of the report.
* **Jump to ID**: Use the searchable dropdown to find a specific ID instantly.
* **View Source**: Click **View Source Video** in the sidebar to open the original footage in a full-screen modal for context.

---

## 5. Understanding the Output
The application generates a "Violation Profile" for every detected non-compliant rider:

| Output Element | Meaning |
| :--- | :--- |
| **Violation ID** | A unique tracking number assigned to a specific rider for the duration of the video. |
| **Confidence Score** | The model's mathematical certainty that the rider is not wearing a helmet. |
| **Close-up Crop** | A zoomed image centered on the detected violation for manual verification. |
| **Context Snapshot** | The full original frame showing the rider's position, surroundings, and lane. |

---

## 6. System Reset
To process a new video:
1. Navigate to the **Sidebar**.
2. Click **Reset & New Upload**. 