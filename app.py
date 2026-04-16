import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="Helmet Violation Detector", layout="wide")

# App States
if "report_data" not in st.session_state:
    st.session_state.report_data = {}
if "processed" not in st.session_state:
    st.session_state.processed = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "model_name" not in st.session_state:
    st.session_state.model_name = ""
if "video_name" not in st.session_state:
    st.session_state.video_name = ""

@st.cache_resource
def load_model(path):
    return YOLO(path)

# App Sidebar
with st.sidebar:
    # Video Upload and Settings Prompt
    if 'video' not in st.session_state:
        st.subheader("Configuration")
        model_choice = st.selectbox("Model Engine", ["yolo26", "yolo11"], index=1)
        st.session_state.model_path = f"{model_choice}n_best.pt"
        st.session_state.model_name = "YOLO11 Nano" if model_choice == "yolo11" else "YOLO26 Nano"

        file = st.file_uploader(
            "Upload Traffic Video (50 MB Limit)",
            type=['mp4', 'avi', 'mov'],
            max_upload_size=50,
            key=f"file_{st.session_state.uploader_key}"
        )
        
        if file:
            st.session_state.video = file.read()
            st.session_state.video_name = file.name 
            st.rerun()
    
    # Video Loaded 
    else:
        # Displaying the parameters used
        st.info(f"**Model:** {st.session_state.model_name}")
        st.write(f"**File:** `{st.session_state.video_name}`")
        
        with st.expander("View Video"):
            st.video(st.session_state.video, muted=True)
        
        st.write("---")
        
        # Reset Button
        if st.button("Clear and Start New Detection", width="stretch", type="primary"):
            st.session_state.processed = False
            st.session_state.report_data = {}
            st.session_state.uploader_key += 1
            st.session_state.current_idx = 0
            if 'video' in st.session_state:
                del st.session_state['video']
            st.rerun()

# Title
st.title("Helmet Violation Detector")

# Processing Video
if 'video' in st.session_state and not st.session_state.processed:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(st.session_state.video)
    
    model = load_model(st.session_state.model_path)
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    history = {} 
    best_instances = {} 
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        frame_count += 1

        # Processing only every third frames to save resources
        if frame_count % 3 != 0: 
            continue 

        results = model.track(frame, persist=True, conf=0.4 if st.session_state.model_name == 'YOLO11 Nano' else 0.401, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            names = results[0].names

            for box, obj_id, cls, conf in zip(boxes, ids, clss, confs):
                if names[cls] != "NoHelmet": 
                    # Skip Objects with Helmets
                    continue
                x1, y1, x2, y2 = map(int, box)
                center = ((x1+x2)//2, (y1+y2)//2)
                
                is_moving = False
                if obj_id in history:
                    dist = np.linalg.norm(np.array(center) - np.array(history[obj_id]))
                    if dist > 3: is_moving = True
                history[obj_id] = center

                if is_moving:
                    if obj_id not in best_instances or conf > best_instances[obj_id]['conf']:
                        annotated_frame = frame.copy()
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        best_instances[obj_id] = {
                            'conf': conf,
                            'image': cv2.cvtColor(annotated_frame[max(0, y1-50):y2+50, max(0, x1-50):x2+50], cv2.COLOR_BGR2RGB),
                            'full_frame': cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        }

        progress_bar.progress(frame_count / total_frames)
        status_text.text(f"Analysing Video... {int((frame_count/total_frames)*100)}%")

    cap.release()
    st.session_state.report_data = best_instances
    st.session_state.processed = True
    status_text.success("Analysis Complete! Review the violations below.")

# Displaying processed result
if st.session_state.processed:
    violations = st.session_state.report_data
    if not violations:
        st.info("No moving helmet violations detected in the provided footage.")
    else:
        st.header(f"{len(violations)} Violations Detected")
        v_ids = sorted(list(violations.keys()))

        # Navigation 
        c_nav1, c_nav2, c_nav3 = st.columns([1, 1, 1])
        with c_nav1:
            if st.button("Previous", width="stretch", disabled=(st.session_state.current_idx == 0)):
                st.session_state.current_idx -= 1
                st.rerun()
        with c_nav2:
            sid = st.selectbox(
                "Violation List", 
                options=v_ids, 
                index=st.session_state.current_idx,
                label_visibility="collapsed"
            )   

            detected_id_index = v_ids.index(sid)
            
            if detected_id_index != st.session_state.current_idx:
                st.session_state.current_idx = detected_id_index
                st.rerun()
    
        with c_nav3:
            if st.button("Next", width="stretch", disabled=(st.session_state.current_idx == len(v_ids)-1)):
                st.session_state.current_idx += 1
                st.rerun()
        st.write("---")

        current_id = v_ids[st.session_state.current_idx]
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"Violation ID: {current_id}")
            st.image(violations[current_id]['image'], width="stretch")
            st.metric("Detection Confidence", f"{violations[current_id]['conf']:.2%}")
        with col2:
            st.subheader("Full Image")
            st.image(violations[current_id]['full_frame'], width="stretch")