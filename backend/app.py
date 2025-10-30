import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Create the Flask App ---
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

def run_face_recognition_algorithm(image_as_array):
    """
    
    """
    print(f"Processing an image with shape: {image_as_array.shape}")
    print("endpoint works")

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
        # 1. Read the image file from memory
        image_data = file.read()
        
        # 2. Convert the raw data to a 1D numpy array of bytes
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
