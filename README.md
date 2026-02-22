<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Tkinter-4B8BBE?style=for-the-badge&logo=python&logoColor=white" alt="Tkinter">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
</p>

# Image Processing with Fourier & Laplacian Filters

*A graphical desktop application for processing and filtering images using spatial and frequency domain techniques, built with Python, OpenCV, and Tkinter.*

## 🌟 Description

This project allows you to explore digital image enhancement, noise generation, and frequency filtering. It provides an interactive GUI to visualize how images behave when transformed into the frequency domain using the **Discrete Fourier Transform (DFT)**.

### Key Features
* **Fourier Transform Visualization:** Compute and display the magnitude spectrum of any loaded image.
* **Noise Generation:** Add structured sinusoidal noise to simulate periodic interference.
* **Notch Filtering:** Manually apply frequency-domain filters to isolate and remove sinusoidal noise from the spectrum.
* **Edge Detection:** Apply the Laplacian operator in the spatial domain for image sharpening and edge extraction.
* **Interactive Blending:** Dynamically blend between original and filtered images.
* **GUI Interface:** Simple Tkinter-based interface for loading, processing, displaying, and saving image results.

These functionalities are ideal for experimenting with computer vision fundamentals, frequency-based filtering, and sharpening techniques.

---

## 📸 Demo

The application provides real-time visualizations of the spatial and frequency domains:

[![Demo GUI](https://raw.githubusercontent.com/toNReverse/image-frequency-filtering/main/demo.png)](https://github.com/toNReverse/image-frequency-filtering/blob/main/demo.png)

---

## 🚀 Installation & Setup

To run this application locally, you need Python installed on your system.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/toNReverse/image-frequency-filtering.git](https://github.com/toNReverse/image-frequency-filtering.git)
   cd image-frequency-filtering
   ```

2. **Install the required dependencies:**
   It is recommended to use a virtual environment. Install `opencv-python` and `numpy`:
   ```bash
   pip install opencv-python numpy
   ```
   *(Note: Tkinter is usually included in standard Python installations. If you face issues on Linux, you may need to install it via your package manager, e.g., `sudo apt-get install python3-tk`).*

3. **Run the application:**
   ```bash
   python image_filter_gui.py
   ```

---

## 📄 License

This project is licensed under the **MIT License**.

For full license details, please refer to the `LICENSE` file included in this repository.
