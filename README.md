# 🧠 TrustMedia: Full-Stack Deepfake Detection Platform

TrustMedia is a comprehensive, full-stack application designed to detect deepfakes in media. It features an entirely custom-built front-end user experience paired with a highly modified, robust machine learning backend.

---

## 🏗️ Project Architecture

The repository is organized as a monorepo to separate the core layers of the application cleanly:

*   **`frontend/`**: An entirely original web interface built from scratch to provide an intuitive user experience for media uploading and real-time analysis visualization.
*   **`backend/`**: A deep learning processing engine utilizing an EfficientNet-B0 architecture fine-tuned for deepfake classification.

---

## ⚙️ Authors & Credits

### 👨‍💻 Maintained & Developed By
*   **Sahil** (GitHub: [@SahilM-CMD](https://github.com/SahilM-CMD))
    *   *Frontend:* 100% original development.
    *   *Backend:* Architecture enhancements, pipeline integration, and heavy code modifications.

### 📊 Acknowledgments
The backend architecture of this project was adapted and heavily modified from the original open-source `DeepfakeDetector` core developed by:
*   [T RAHUL SINGH](https://github.com/TRahulsingh)
*   [Mallikarjun Macherla](https://github.com/Mallikarjun-Macherla)
*   [Sainath](https://github.com/sainathch45/)

---

## 🚀 Getting Started

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   
Install the required Python dependencies:
pip install -r requirements.txt
Run the application:
python web-app.py
Frontend Setup
Navigate to the frontend directory:
cd frontend