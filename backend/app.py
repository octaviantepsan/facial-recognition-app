import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Create the Flask App ---
# This is the core of your Python backend
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing)
# This allows your HTML/JS frontend to make requests to this backend
CORS(app)

# --- Your Scientific/Face Recognition Code ---
# We'll put all your math-heavy code here.
# To keep this example simple, our "algorithm" will just
# get the image dimensions and pretend to find faces.

def run_face_recognition_algorithm(image_as_array):
    """
    This is where you'd use libraries like opencv, dlib,
    face_recognition, or your own scientific algorithms.
    
    The input 'image_as_array' is a NumPy array, just
    like you wanted!
    """
    print(f"Processing an image with shape: {image_as_array.shape}")
    
    # Get image dimensions
    height, width, channels = image_as_array.shape
    
    # ---
    # FAKE ALGORITHM:
    # Let's pretend we found two faces.
    # In your real app, this would be the output of your model.
    # ---
    faces_found = [
        {
            "name": "Alex",
            "confidence": 0.98,
            "box": [int(width * 0.1), int(height * 0.2), int(width * 0.3), int(height * 0.5)] # [x1, y1, x2, y2]
        },
        {
            "name": "Unknown",
            "confidence": 0.75,
            "box": [int(width * 0.6), int(height * 0.3), int(width * 0.8), int(height * 0.6)]
        }
    ]
    
    # We return the results as a Python dictionary.
    # Flask will automatically turn this into JSON for the frontend.
    return {
        "status": "success",
        "image_dimensions": {
            "height": height,
            "width": width,
            "channels": channels
        },
        "faces_found": faces_found
    }

# --- API Endpoint ---
# This defines a "route" or a URL that the frontend can call.
# We'll set it up at http://localhost:5000/process_image

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
        # Read the image file from memory
        image_data = file.read()
        # Open the image using Pillow (PIL)
        image = Image.open(io.BytesIO(image_data))

        # --- THIS IS THE KEY PART ---
        # Convert the image to a NumPy array, just as you
        # do in your scientific work.
        image_array = np.array(image)
        # ---------------------------

        # If the image has an alpha channel (like PNGs), let's keep it simple
        # and just take the first 3 (RGB) channels.
        if image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]

        # Run your actual processing function
        results = run_face_recognition_algorithm(image_array)
        
        # Send the results back to the frontend
        return jsonify(results)

    except Exception as e:
        # Handle any errors during processing
        print(f"An error occurred: {e}")
        return jsonify({"error": "Failed to process image", "message": str(e)}), 500

# --- Run the Server ---
if __name__ == "__main__":
    # Starts the web server on http://localhost:5000
    # The debug=True flag automatically reloads the server when you save changes.
    print("Starting Python backend server at http://localhost:5000")
    app.run(debug=True, port=5000)
