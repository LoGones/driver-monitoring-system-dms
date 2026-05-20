# Driver Monitoring System (DMS)
> A real-time computer vision system to monitor driver state, detect distractions, drowsiness, and features non-intrusive heart rate (BPM) estimation via Remote Photoplethysmography (rPPG) and FastICA.
> 
> built with Python and MediaPipe, using a standard webcam.

## Overview
This project implements a comprehensive Driver Monitoring System (DMS) entirely in Python. By leveraging **Google MediaPipe** for robust 3D facial landmark tracking and **Independent Component Analysis (FastICA)** for Remote Photoplethysmography (rPPG), the system enhances road safety by providing real-time, non-intrusive alerts for critical driver conditions.

## Key Features
- **👀 Distraction Detection (Owl & Lizard):**
  - *Owl Distraction:* Tracks overall head rotation by monitoring the spatial displacement of the nose tip relative to the frame center. Triggers short (cumulative) and long continuous distraction warnings.
  - *Lizard Distraction:* Monitors gaze and iris offsets to detect when the driver looks away using only their eyes, without moving their head.
- **😴 Drowsiness Detection:**
  - Continuously calculates the Eye Aspect Ratio (EAR) using specific eyelid landmarks.
  - Issues prioritized warnings for *Microsleep* (eyes closed $\ge$ 4s) and critical *Sleep* (eyes closed $\ge$ 7s).
  - Features morphological threshold calibration to prevent false positives across different users and facial structures.
- **❤️ Real-Time Heart Rate (BPM) Monitor:**
  - Dynamically extracts raw RGB color signals from highly vascularized facial regions (forehead and cheekbones).
  - Utilizes the **FastICA** algorithm to isolate the blood volume pulse from ambient lighting noise, estimating the driver's heart rate in the physiological 45-180 BPM range.

## Project Architecture
The architecture follows a clean, modular design separated into three core components:
1. `runDMS.py`: The main orchestrator. Handles webcam stream acquisition, state machine logic (temporal buffers and timers), and real-time UI rendering via OpenCV.
2. `mediapipe_module.py`: The feature extraction engine. Encapsulates the `FaceAnalyzer` class to process MediaPipe Face Mesh data, calculating geometry (EAR, gaze vectors) and extracting dynamic ROIs.
3. `ICA_module.py`: The physiological signal processor. Implements custom FastICA to decompose mixed RGB signals, analyze the Power Spectral Density (PSD), and isolate the cardiac frequency.

## ⚙️ Installation
1. Clone this repository:
2. Install the required Python dependencies. It is highly recommended to use a virtual environment:
  ```Bash
  pip install opencv-python mediapipe numpy
  ```

## Usage
Run the main script to start the monitoring system. Ensure your environment is reasonably well-lit and your webcam is active
  ```Bash
  python runDMS.py
  ```

### Note: The system requires a few seconds upon startup to populate the signal buffers before displaying the first BPM reading.

## Operational Robustness
Empirical testing demonstrates that this DMS exhibits substantial resilience to ambient artifacts and partial facial occlusions (e.g., wearing hats or standard eyewear). The baseline EAR evaluation achieves cross-user invariance, meaning the system seamlessly generalizes to new drivers without requiring individual manual recalibration.

Author: Luca Gonella.
