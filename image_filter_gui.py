import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk

# Global images
image = None
original_image = None

# Noise parameters for noise 1
NOISE_AMPLITUDE = 50
NOISE_FREQUENCY = 5 / 512
NOISE_PHASE = 0

# Noise parameters for noise 2
NOISE2_AMPLITUDE = 30
NOISE2_FREQUENCY = 8 / 512
NOISE2_PHASE = np.pi / 4

def load_image():
    """Load an image, resize it to 512x512, and display it in the GUI."""
    global image, original_image
    path = filedialog.askopenfilename()
    if path:
        image = cv2.resize(cv2.imread(path), (512, 512))
        original_image = image.copy()
        show_image(image)

def show_image(img):
    """Convert the image to RGB format and display it in the GUI."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(img))
    panel.config(image=imgtk)
    panel.image = imgtk

def apply_fourier_transform():
    """Apply the Fourier Transform and display the magnitude spectrum."""
    if image is None:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    spectrum = 20 * np.log(np.abs(fshift) + 1)
    spectrum = cv2.normalize(spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    show_image(spectrum)
    cv2.imwrite("spettro_fourier.png", spectrum)

def apply_laplacian_filter():
    """Apply Laplacian filter with a user-defined kernel size."""
    global image
    if image is None:
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = simpledialog.askinteger("Kernel", "Enter odd kernel size:", minvalue=1)
    k = k + 1 if k % 2 == 0 else k
    laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=k)
    image = cv2.convertScaleAbs(laplacian)
    show_image(image)

def add_sinusoidal_noise():
    """Add diagonal sinusoidal noise to the luminance channel (Y)."""
    global image
    if image is None:
        return
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    height, width = Y.shape
    X, Y_grid = np.meshgrid(np.arange(width), np.arange(height))
    noise = NOISE_AMPLITUDE * np.sin(2 * np.pi * NOISE_FREQUENCY * (X + Y_grid) + NOISE_PHASE)
    Y_noisy = np.clip(Y.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(cv2.merge([Y_noisy, Cr, Cb]), cv2.COLOR_YCrCb2BGR)
    show_image(image)

def remove_noise_frequency():
    """Remove diagonal sinusoidal noise using a notch filter in the frequency domain."""
    global image
    if image is None:
        return
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    fshift = np.fft.fftshift(np.fft.fft2(Y))
    rows, cols = Y.shape
    mask = np.ones((rows, cols), np.float32)
    notch_size, attenuation, offset = 5, 0.3, 5
    crow, ccol = rows // 2, cols // 2
    for dx in [-offset, offset]:
        for dy in [-offset, offset]:
            mask[max(crow+dy-notch_size,0):min(crow+dy+notch_size,rows), 
                 max(ccol+dx-notch_size,0):min(ccol+dx+notch_size,cols)] = attenuation
    filtered_f = np.fft.ifftshift(fshift * mask)
    Y_filtered = np.real(np.fft.ifft2(filtered_f))
    Y_filtered = normalize_to_original(Y, Y_filtered)
    blend_slider(Y, Y_filtered, Cr, Cb)

def normalize_to_original(Y_orig, Y_filtered):
    """Normalize the filtered image to match the original luminance statistics."""
    mean_orig, std_orig = np.mean(Y_orig), np.std(Y_orig)
    mean_filt, std_filt = np.mean(Y_filtered), np.std(Y_filtered)
    if std_filt > 1e-5:
        Y_filtered = (Y_filtered - mean_filt) * (std_orig / std_filt) + mean_orig
    return np.clip(Y_filtered, 0, 255)

def blend_slider(Y, Y_filtered, Cr, Cb):
    """Create an interactive slider to blend the original and filtered image."""
    def update_blend(val):
        alpha = int(val) / 100.0
        blended = alpha * Y_filtered + (1 - alpha) * Y
        blended = cv2.GaussianBlur(blended.astype(np.float32), (3, 3), 0)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(cv2.merge([blended, Cr, Cb]), cv2.COLOR_YCrCb2BGR)
        show_image(img)

    win = tk.Toplevel()
    win.title("Filtered Blend")
    tk.Label(win, text="Filtered Blend").pack()
    slider = tk.Scale(win, from_=0, to=100, orient=tk.HORIZONTAL, length=300, command=update_blend)
    slider.set(70)
    slider.pack()

def add_sinusoidal_noise_2():
    """Add horizontal sinusoidal noise to the luminance channel (Y)."""
    global image
    if image is None:
        return
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    height, width = Y.shape
    X = np.arange(width)
    noise = NOISE2_AMPLITUDE * np.sin(2 * np.pi * NOISE2_FREQUENCY * X + NOISE2_PHASE)
    noise = np.tile(noise, (height, 1))
    Y_noisy = np.clip(Y.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(cv2.merge([Y_noisy, Cr, Cb]), cv2.COLOR_YCrCb2BGR)
    show_image(image)

def remove_noise_frequency_2():
    """Remove horizontal sinusoidal noise using a notch filter in the frequency domain."""
    global image
    if image is None:
        return
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    fshift = np.fft.fftshift(np.fft.fft2(Y))
    rows, cols = Y.shape
    mask = np.ones((rows, cols), np.float32)
    notch_size, attenuation = 5, 0.3
    offset_x = int(NOISE2_FREQUENCY * cols)
    crow, ccol = rows // 2, cols // 2
    for dx in [-offset_x, offset_x]:
        mask[max(crow-notch_size,0):min(crow+notch_size,rows), 
             max(ccol+dx-notch_size,0):min(ccol+dx+notch_size,cols)] = attenuation
    filtered = np.real(np.fft.ifft2(np.fft.ifftshift(fshift * mask)))
    filtered = np.clip(normalize_to_original(Y, filtered), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(cv2.merge([filtered, Cr, Cb]), cv2.COLOR_YCrCb2BGR)
    show_image(image)

def reset_image():
    """Restore the original loaded image."""
    global image
    if original_image is not None:
        image = original_image.copy()
        show_image(image)

def save_image():
    """Save the currently displayed image to disk."""
    if image is None:
        return
    path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
    if path:
        cv2.imwrite(path, image)

# ---------------- GUI ----------------

window = tk.Tk()
window.title("Python + OpenCV GUI")

# Main buttons
btn_load = tk.Button(window, text="Load", command=load_image)
btn_load.grid(row=0, column=0)

btn_fourier = tk.Button(window, text="Fourier", command=apply_fourier_transform)
btn_fourier.grid(row=0, column=1)

btn_laplacian = tk.Button(window, text="Laplacian", command=apply_laplacian_filter)
btn_laplacian.grid(row=0, column=2)

btn_add_noise = tk.Button(window, text="Add Noise", command=add_sinusoidal_noise)
btn_add_noise.grid(row=0, column=3)

btn_remove_noise = tk.Button(window, text="Remove Noise", command=remove_noise_frequency)
btn_remove_noise.grid(row=0, column=4)

btn_add_noise2 = tk.Button(window, text="Add Noise 2", command=add_sinusoidal_noise_2)
btn_add_noise2.grid(row=0, column=7)

btn_remove_noise2 = tk.Button(window, text="Remove Noise 2", command=remove_noise_frequency_2)
btn_remove_noise2.grid(row=0, column=8)

btn_reset = tk.Button(window, text="Reset", command=reset_image)
btn_reset.grid(row=0, column=9)

btn_save = tk.Button(window, text="Save", command=save_image)
btn_save.grid(row=0, column=10)

# Image display panel
panel = tk.Label(window)
panel.grid(row=1, column=0, columnspan=11)

# Start the GUI loop
window.mainloop()
