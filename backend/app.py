import numpy as np
import cv2
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Create the Flask App ---
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

def run_face_recognition_algorithm(image_as_array, normType='2'):
    if normType == 'cos':
        distances = np.array([calcNorm(train_images[i], image_as_array, normType) for i in range(len(train_images))])
    else:
        if normType == '1':
            ord = 1
        elif normType == 'inf':
            ord = np.inf
        elif normType == '2':
            ord = 2
        else:
            print(f"Warning: Unknown norm type '{normType}', defaulting to Euclidean (L2)")
            ord = 2
        
        distances = np.linalg.norm(train_images - image_as_array, ord=ord, axis=1)
    
    nearest_idx = int(np.argmin(distances))
    
    print("---------------------")
    print(image_as_array)
    print("---------------------")
    print(f"Processing an image with shape: {image_as_array.shape}")
    print("endpoint works")
    
    print(f"Nearest index in training set: {nearest_idx}")
    
    return {"nearest_idx": nearest_idx}

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
    
    base_path = r"C:\Octavian\github\facial-recognition\facial-recognition-app\assets\attfaces"

    for class_dir in os.listdir(base_path):
        class_path = os.path.join(base_path, class_dir)
        
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

    print(f"train_images shape: {train_images.shape}, test_images shape: {test_images.shape}")


def calcNorm(train_sample, test_sample, normType='2'):
    """Calculate distance between two samples using specified norm.
    
    Args:
        train_sample: Single training image (flattened)
        test_sample: Single test image (flattened)
        normType: Type of norm ('1'=Manhattan, '2'=Euclidean, 'inf'=Chebyshev, 'cos'=Cosine)
    """
    if normType == '1':
        return np.linalg.norm(train_sample - test_sample, ord=1)
    elif normType == '2':
        return np.linalg.norm(train_sample - test_sample, ord=2)
    elif normType == 'inf':
        return np.linalg.norm(train_sample - test_sample, ord=np.inf)
    elif normType == 'cos':
        return 1 - np.dot(train_sample, test_sample) / (np.linalg.norm(train_sample) * np.linalg.norm(test_sample))
    else:
        raise ValueError(f"Unknown norm type: {normType}")

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
        # If the user didn't send an 'image' file, return an error
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    
    try:
        # read the image file from memory
        image_data = file.read()
        
        # convert the raw data to a 1D numpy array of bytes
        nparr = np.frombuffer(image_data, np.uint8)
        
        image_array_uint8 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        # Check if image was decoded successfully
        if image_array_uint8 is None:
            return jsonify({"error": "Failed to decode image. Is it a valid image file?"}), 400

        image_array_float = image_array_uint8.astype(np.float32)
        results = run_face_recognition_algorithm(image_array_float)
        
        return jsonify(results)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to process image", "message": str(e)}), 500

# --- Run the Server ---
if __name__ == "__main__":
    # Starts the web server on http://localhost:5000
    # The debug=True flag automatically reloads the server when you save changes.
    print("Starting Python backend server at http://localhost:5000")
    app.run(debug=True, port=5000)
