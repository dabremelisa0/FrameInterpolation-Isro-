# AI-Powered Frame Interpolation for Satellite Imagery

This project focuses on enhancing satellite imagery visualization by generating intermediate frames using AI-based **Frame Interpolation** techniques. It utilizes the **FILM (Frame Interpolation for Large Motion)** model to improve temporal smoothness between satellite images obtained from the **MOSDAC** dataset via **WMS (Web Map Service)**. The results are displayed on an interactive web interface using **OpenLayers**.

## 🚀 Key Features

- 🌍 **Satellite Imagery via WMS** from [MOSDAC](https://www.mosdac.gov.in/)
- 🧠 **FILM Model** for AI-based frame interpolation
- 📈 **88% Accuracy** in temporal interpolation evaluation
- 🗺️ Interactive visualization with **OpenLayers**
- 🌐 Responsive web deployment for public access

---

## 📚 Technologies Used

| Component | Stack |
|----------|-------|
| **Frontend** | HTML, JavaScript, OpenLayers |
| **Backend/Processing** | Python, TensorFlow (FILM Model), Flask (optional API support) |
| **Data Source** | Satellite data from MOSDAC via WMS |
| **Model** | FILM (Frame Interpolation for Large Motion) |

---

## 🎥 What is Frame Interpolation?

Frame Interpolation is a technique used to generate intermediate frames between two given frames. This is highly useful in satellite imagery to:

- Create smoother time-lapse visualizations
- Improve perceptual continuity in environmental monitoring
- Analyze motion patterns over time

---

## 🧠 FILM Model (Google Research)

The **FILM model** is a state-of-the-art deep learning model designed for large-motion video frame interpolation. It uses:

- **U-Net based encoder-decoder architecture**
- **Bidirectional optical flow estimation**
- **Feature warping & fusion**

**Key Benefits:**

- Handles complex and large motion transitions
- High-quality temporal interpolation
- No need for fine-tuning on the dataset

### Resources:
- [FILM Research Paper](https://arxiv.org/abs/2202.04901)
- [Original TensorFlow Implementation](https://github.com/google-research/frame-interpolation)

---

## 🗺️ OpenLayers for Visualization

**OpenLayers** is an open-source JavaScript library for displaying dynamic map data in web browsers. In this project:

- It overlays interpolated frames on base satellite maps
- Allows zooming, panning, and timeline playback
- Works seamlessly with WMS layers from MOSDAC

### Live Features:

- Time navigation bar
- Base layer switching
- Overlaying interpolated vs original frames for comparison

---

## 🛠️ How It Works

1. **Data Fetching:** Download satellite images using WMS from MOSDAC.
2. **Preprocessing:** Resize and align frames, prepare them for model input.
3. **Frame Interpolation:** Feed pairs of consecutive frames to the FILM model to generate intermediate frames.
4. **Postprocessing:** Save outputs and load them into the OpenLayers frontend for interaction.
5. **Deployment:** Host as a static or Flask-based app with accessible URL.

---

## 📊 Results

- Interpolation Accuracy: **88%** (PSNR/SSIM-based evaluation)
- Performance: ~30 FPS on NVIDIA GPU
- Use Case: Real-time frame enhancement for rainfall/cloud movement

---

## 📁 Folder Structure

