import numpy as np
import cv2
import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Create the Flask App ---
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

def run_face_recognition_algorithm(image_as_array, normType='2'):
    distances = calcDist(image_as_array, normType)
    
    nearest_idx = int(np.argmin(distances))
    
    print("---------------------")
    print(image_as_array)
    print("---------------------")
    print(f"Processing an image with shape: {image_as_array.shape}")
    print(f"Nearest index in training set: {nearest_idx}")
    
    return nearest_idx

def load_images_and_paths():
    global train_images, test_images
    """
    Load training and testing images from the specified base path.
    
    Args:
        base_path: Path to the dataset directory.

    Returns:
        train_images: List of training images.
        test_images: List of testing images.
    """
    
    base_path_laptop = r"C:\Octavian\github\facial-recognition\facial-recognition-app\assets\attfaces"
    base_path_pc = r"D:\OCTAVIAN\github\facial-recognition-app\assets\attfaces"

    for class_dir in os.listdir(base_path_pc):
        class_path = os.path.join(base_path_pc, class_dir)
        
        if not os.path.isdir(class_path):
            continue

        images = sorted(os.listdir(class_path))
        train_imgs = images[:8]
        test_imgs = images[-2:]

        for img_name in train_imgs:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            train_images.append(img.flatten().astype(np.float32))

        for img_name in test_imgs:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            test_images.append(img.flatten().astype(np.float32))

    train_images = np.array(train_images)
    test_images = np.array(test_images)

    print("------------------------------")
    print(f"train_images shape: {train_images.shape}, test_images shape: {test_images.shape}")
    print("------------------------------")

def calcDist(test_sample, normType='2'):
    """
        normType: Type of norm ('1'=Manhattan, '2'=Euclidean, 'inf'=Chebyshev, 'cos'=Cosine)
    """
    
    if normType == 'cos':
        dot_products = np.dot(train_images, test_sample)
    
        train_norms = np.linalg.norm(train_images, axis=1)
        test_norm = np.linalg.norm(test_sample)
        
        epsilon = 1e-10
        cosine_similarity = dot_products / ((train_norms * test_norm) + epsilon)
        
        return 1 - cosine_similarity
    
    if normType == '1':
        ord = 1
    elif normType == 'inf':
        ord = np.inf
    elif normType == '2':
        ord = 2
        
    distances = np.linalg.norm(train_images - test_sample, ord=ord, axis=1)
    
    return distances

def prepareImgToSend(matched_img_index):
    matched_img = train_images[matched_img_index]
    matched_img_2d = matched_img.reshape(112, 92)
    matched_image_uint8 = matched_img_2d.astype(np.uint8)
    
    # encode the 2D array as a PNG in memory
    is_success, buffer = cv2.imencode(".png", matched_image_uint8)
    
    # convert that in-memory PNG to a Base64 text string
    b64_string = base64.b64encode(buffer).decode("utf-8")
    
    return b64_string

train_images = []
test_images = []

load_images_and_paths()

# --- API Endpoint ---

@app.route("/process_image", methods=["POST"]) 
def handle_image_processing():
    """
    This function is called when the frontend sends an image
    to the /process_image URL.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    
    try:
        # read the image file from memory
        image_data = file.read()
        
        # convert the raw data to a 1D numpy array of bytes
        nparr = np.frombuffer(image_data, np.uint8)
        
        image_array_uint8 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image_array_uint8 is None:
            return jsonify({"error": "Failed to decode image. Is it a valid image file?"}), 400

        image_array_float = image_array_uint8.flatten().astype(np.float32)
        matched_img_index = run_face_recognition_algorithm(image_array_float)
        
        img_as_string = prepareImgToSend(matched_img_index)
        
        return jsonify({
            "matched_img_index": matched_img_index,
            "image_b64": img_as_string,
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to process image", "message": str(e)}), 500

TARGET_SHAPE_2D = (112, 92) 

@app.route('/preview', methods=['POST'])
def preview_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    try:
        # Read the .pgm file data
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image_array_uint8 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image_array_uint8 is None:
            return jsonify({"error": "Failed to decode PGM image"}), 400

        # --- This is just for preview, so we send it back as PNG ---
        
        # Resize it so the preview looks right
        image_resized = cv2.resize(image_array_uint8, (TARGET_SHAPE_2D[1], TARGET_SHAPE_2D[0]))

        # Encode the resized 2D array as a PNG in memory
        is_success, buffer = cv2.imencode(".png", image_resized)
        
        # Convert that in-memory PNG to a Base64 text string
        b64_string = base64.b64encode(buffer).decode("utf-8")

        # Send the PNG string back to the frontend
        return jsonify({"image_b64": b64_string})

    except Exception as e:
        print(f"Preview error: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

# --- Run the Server ---
if __name__ == "__main__":
    # Starts the web server on http://localhost:5000
    # The debug=True flag automatically reloads the server when you save changes.
    print("Starting Python backend server at http://localhost:5000")
    app.run(debug=True, port=5000)
