import streamlit as st
import cv2
import json
import torch
import numpy as np
import av
import time
from PIL import Image
from typing import NamedTuple, List
from singleinference_yolov7 import SingleInference_YOLOV7


def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets.get("password", "pass"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True


if check_password():
    st.set_page_config(
        page_title="Live Webcam PPE Detection - Securade",
        page_icon="👷",
        layout="wide"
    )

    st.title("👷 Live Webcam PPE Detection")
    st.markdown("Deteksi penggunaan Alat Pelindung Diri (APD/PPE) seperti **Helm (Hardhat)**, **Rompi (Safety Vest)**, dan **Masker** secara langsung dari kamera laptop/webcam.")

    # Load configuration
    CONFIG_FILE = './configs/default.json'
    try:
        model_from_config = json.load(open(CONFIG_FILE))
        MODEL_WEIGHTS = model_from_config.get('model', 'modelzoo/safety.pt')
    except Exception:
        MODEL_WEIGHTS = 'modelzoo/safety.pt'

    IMG_SIZE = 640

    # Sidebar: Model Parameters & Device Info
    st.sidebar.header("⚙️ Pengaturan Model")
    confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.25, 0.05)
    overlap_threshold = st.sidebar.slider("Overlap (IOU) Threshold:", 0.0, 1.0, 0.45, 0.05)

    device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.success(f"🚀 Akselerasi: {device_str}\n({gpu_name})")
    else:
        st.sidebar.info(f"💻 Akselerasi: {device_str}")

    # Load YOLOv7 model
    @st.cache_resource
    def load_detector(weights, conf, iou):
        device_i = "0" if torch.cuda.is_available() else "cpu"
        model = SingleInference_YOLOV7(
            img_size=IMG_SIZE,
            path_yolov7_weights=weights,
            device_i=device_i,
            conf_thres=conf,
            iou_thres=iou
        )
        model.load_model()
        return model

    yolov7_detector = load_detector(MODEL_WEIGHTS, confidence_threshold, overlap_threshold)
    yolov7_detector.conf_thres = confidence_threshold
    yolov7_detector.iou_thres = overlap_threshold

    # PPE Policy Configuration
    st.subheader("📋 Aturan Keselamatan PPE")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ Standar Kepatuhan APD (Warna Hijau):**")
        hardhats = st.checkbox("Wajib Helm (Wearing Hardhat)", value=True, key="live_hardhats")
        vests = st.checkbox("Wajib Rompi (Wearing Vest)", value=True, key="live_vests")
        masks = st.checkbox("Wajib Masker (Wearing Mask)", value=False, key="live_masks")

    with col2:
        st.markdown("**⚠️ Pelanggaran APD (Warna Merah):**")
        no_hardhats = st.checkbox("Tanpa Helm (Not wearing Hardhat)", value=True, key="live_no_hardhats")
        no_vests = st.checkbox("Tanpa Rompi (Not wearing Vest)", value=False, key="live_no_vests")
        no_masks = st.checkbox("Tanpa Masker (Not wearing Mask)", value=False, key="live_no_masks")

    policy = {
        "type": "ppe_detection",
        "hardhats": hardhats,
        "vests": vests,
        "masks": masks,
        "no_hardhats": no_hardhats,
        "no_vests": no_vests,
        "no_masks": no_masks
    }

    st.download_button(
        label="💾 Unduh Aturan APD (JSON Policy)",
        data=json.dumps(policy, indent=4),
        file_name="ppe_webcam_policy.json",
        mime="application/json"
    )

    st.divider()

    mode_choice = st.radio(
        "Pilih Metode Kamera:",
        [
            "🎥 Live Stream Native (OpenCV - Sangat Cepat & Stabil)",
            "🌐 Live Stream WebRTC (In-Browser Stream)",
            "📸 Snapshot Foto (Camera Input)"
        ],
        index=0
    )

    # MODE 1: NATIVE OPENCV STREAM
    if mode_choice == "🎥 Live Stream Native (OpenCV - Sangat Cepat & Stabil)":
        st.subheader("Live Stream OpenCV")
        col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
        with col_c1:
            cam_idx = st.number_input("Index Kamera:", min_value=0, max_value=5, value=0, step=1)
        with col_c2:
            frame_skip = st.slider("Frame Skip:", 1, 5, 2)
        with col_c3:
            run_live = st.checkbox("🔴 Mulai Deteksi Kamera Laptop", key="run_native_live")

        frame_placeholder = st.image([])
        status_placeholder = st.empty()
        table_placeholder = st.empty()

        if run_live:
            # Open camera cleanly
            cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_MSMF)
            if not cap.isOpened():
                cap = cv2.VideoCapture(int(cam_idx))

            if not cap.isOpened():
                st.error("❌ Kamera tidak dapat dibuka. Pastikan tidak ada aplikasi lain (Zoom, Teams, Camera App) yang sedang mengakses kamera.")
            else:
                status_placeholder.info("⏳ Memulai feed kamera laptop...")
                frame_count = 0
                try:
                    while run_live:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.05)
                            continue

                        frame_count += 1
                        if frame_count % frame_skip == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            annotated_img = yolov7_detector.detect_ppe(
                                frame_rgb, hardhats, vests, masks, no_hardhats, no_vests, no_masks
                            )
                            frame_placeholder.image(annotated_img, channels="RGB", use_column_width=True)

                            # Detection info
                            detections = []
                            person_count = 0
                            violations = 0
                            for item in yolov7_detector.predicted_bboxes_PascalVOC:
                                label = str(item[0])
                                prob = round(100 * float(item[-1]), 1)
                                detections.append({"Objek": label, "Confidence (%)": prob})
                                if label == "Person":
                                    person_count += 1
                                if "NO-" in label:
                                    violations += 1

                            if person_count == 0:
                                status_placeholder.info("ℹ️ Tidak ada orang terdeteksi.")
                            elif violations > 0:
                                status_placeholder.error(f"🚨 PERINGATAN: Terdeteksi {person_count} Orang dengan {violations} Pelanggaran APD!")
                            else:
                                status_placeholder.success(f"✅ AMAN: Terdeteksi {person_count} Orang mematuhi standar APD.")

                            if detections:
                                table_placeholder.table(detections)

                finally:
                    cap.release()

    # MODE 2: WebRTC In-Browser Stream
    elif mode_choice == "🌐 Live Stream WebRTC (In-Browser Stream)":
        st.subheader("Live WebRTC Stream")
        st.caption("Klik **START** di bawah untuk memulai streaming WebRTC.")
        from streamlit_webrtc import webrtc_streamer, RTCConfiguration

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            annotated = yolov7_detector.detect_ppe(
                img_rgb, hardhats, vests, masks, no_hardhats, no_vests, no_masks
            )
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

        webrtc_streamer(
            key="ppe-webrtc",
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # MODE 3: Snapshot Photo
    else:
        st.subheader("Ambil Foto dari Kamera")
        captured_image = st.camera_input("Arahkan kamera laptop Anda dan klik 'Take Photo':")

        if captured_image is not None:
            pil_img = Image.open(captured_image)
            img_arr = np.array(pil_img)

            # Perform detection
            result_img = yolov7_detector.detect_ppe(
                img_arr, hardhats, vests, masks, no_hardhats, no_vests, no_masks
            )

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(img_arr, caption="Foto Asli", use_column_width=True)
            with col_res2:
                st.image(result_img, caption="Hasil Analisis APD / PPE", use_column_width=True)

            # Details
            detected_items = []
            violations = 0
            for item in yolov7_detector.predicted_bboxes_PascalVOC:
                lbl = str(item[0])
                prob = round(100 * float(item[-1]), 1)
                detected_items.append({"Label": lbl, "Confidence (%)": prob})
                if "NO-" in lbl:
                    violations += 1

            if violations > 0:
                st.error(f"⚠️ Ditemukan {violations} Pelanggaran APD pada foto ini.")
            else:
                st.success("✅ Semua orang dalam foto memenuhi kepatuhan APD.")

            if detected_items:
                st.write("**Rincian Objek Terdeteksi:**")
                st.table(detected_items)
