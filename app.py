from flask import Flask, request, render_template, redirect, url_for, send_file
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove
from skimage import exposure
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Function Definitions
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(image, kernel_size=15):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_vignette(image):
    rows, cols = image.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    vignette = np.copy(image)
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    return vignette

def apply_adaptive_cartoon(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 10)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blurred = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    return sketch

def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

def preprocess_image_for_removal(image):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_pil = image_pil.filter(ImageFilter.SHARPEN)
    return np.array(image_pil)

def remove_background(image):
    preprocessed_image = preprocess_image_for_removal(image)
    result_pil = remove(preprocessed_image)
    result_image = np.array(result_pil)
    return cv2.cvtColor(result_image, cv2.COLOR_RGBA2BGR)

def apply_edge_detection(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def process_image(image_path, effect, **kwargs):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not loaded properly")
    if effect == 'Black & White':
        result = to_grayscale(img)
    elif effect == 'Blur':
        result = apply_gaussian_blur(img, 15)
    elif effect == 'Vignette':
        result = apply_vignette(img)
    elif effect == 'Cartoon':
        result = apply_adaptive_cartoon(img)
    elif effect == 'Sketch':
        result = apply_sketch(img)
    elif effect == 'Flip':
        result = flip_image(img, 1)
    elif effect == 'Remove Background':
        result = remove_background(img)
    elif effect == 'Edge Detection':
        result = apply_edge_detection(img)
    else:
        result = img

    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            effect = request.form.get('effect')
            kwargs = {key: request.form.get(key) for key in ['blur_intensity', 'flip_code', 'darken_factor', 'glow_intensity', 'r_balance', 'g_balance', 'b_balance']}
            result = process_image(filepath, effect, **kwargs)
            result_filename = 'result_' + filename
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_filepath, result)
            return redirect(url_for('uploaded_file', filename=result_filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('uploaded.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
