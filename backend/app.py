import numpy as np
import cv2
import os
import base64
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy import stats

# --- Create the Flask App ---
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

def run_nn_algorithm(image_as_array, normType='2'):
    distances = calcDist(image_as_array, normType)
    
    nearest_idx = int(np.argmin(distances))
    
    return nearest_idx

def run_knn_algorithm(image_as_array, k=1, normType='2'):
    """
    Finds the K-nearest neighbors and returns the most common person label.
    Uses index // 8 to determine the person.
    """
    
    distances = calcDist(image_as_array, normType)
    
    k_nearest_indices = np.argsort(distances)[:k]
    
    k_person_labels = [(idx // 8) + 1 for idx in k_nearest_indices]
    
    most_common_label = stats.mode(k_person_labels)[0]
    
    single_nearest_idx = int(k_nearest_indices[0])
    
    return {
        "person_label": int(most_common_label),
        "nearest_idx": single_nearest_idx
    }

def load_images_and_paths():
    global train_images, test_images
    
    # Initialize as lists
    train_images_list = []
    test_images_list = []
    
    base_path_laptop = r"C:\Octavian\github\facial-recognition\facial-recognition-app\assets\attfaces"
    base_path_pc = r"D:\OCTAVIAN\github\facial-recognition-app\assets\attfaces"

    base_path_to_use = os.path.normpath(base_path_pc)
    if not os.path.exists(base_path_to_use):
        base_path_to_use = os.path.normpath(base_path_laptop)
        if not os.path.exists(base_path_to_use):
            print("ERROR: Neither asset path exists.")
            return

    try:
        dir_list = sorted(os.listdir(base_path_to_use), key=lambda x: int(x[1:]) if x.startswith('s') else 0)
    except Exception as e:
        print(f"Failed to sort directories, are they named 's1', 's2'...? Error: {e}")
        return

    for class_dir in dir_list:
        class_path = os.path.join(base_path_to_use, class_dir)
        
        if not os.path.isdir(class_path) or not class_dir.startswith('s'):
            continue

        images = sorted(os.listdir(class_path), key=lambda x: int(x.split('.')[0]))
        train_imgs = images[:8]
        test_imgs = images[-2:]

        for img_name in train_imgs:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                train_images_list.append(img.flatten().astype(np.float32))

        for img_name in test_imgs:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                test_images_list.append(img.flatten().astype(np.float32))
                
    train_images = np.array(train_images_list)
    test_images = np.array(test_images_list)

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

def get_recognition_results(image_array_float, algorithm, k, normType):
    """
    Runs the selected algorithm and returns a dictionary
    ready to be sent to the frontend.
    """
    if algorithm == 'nn':
        nearest_idx = run_nn_algorithm(image_array_float, normType=normType)
        person_label = (nearest_idx // 8) + 1
        
        img_as_string = prepareImgToSend(nearest_idx)
        algo_name = "NN (Nearest Neighbor)"

        return {
            "algorithm": algo_name,
            "person_label": person_label,
            "nearest_idx": nearest_idx,
            "image_b64": img_as_string,
        }

    else: # (algorithm == 'knn')
        results_dict = run_knn_algorithm(image_array_float, k=k, normType=normType)
        person_label = results_dict["person_label"]
        nearest_idx = results_dict["nearest_idx"]
        
        first_index_of_winner = (person_label - 1) * 8
        img_as_string = prepareImgToSend(first_index_of_winner)
        algo_name = f"K-NN (k={k})"

        # make K-NN for k=1 work same as NN
        if k == 1:
            img_as_string = prepareImgToSend(nearest_idx)
            algo_name = f"K-NN (k=1)"
        else:
            first_index_of_winner = (person_label - 1) * 8
            img_as_string = prepareImgToSend(first_index_of_winner)
            algo_name = f"K-NN (k={k})"

        return {
            "algorithm": algo_name,
            "person_label": person_label,
            "nearest_idx": nearest_idx,
            "image_b64": img_as_string,
        }

train_images = []
test_images = []
TARGET_SHAPE_2D = (112, 92) 

load_images_and_paths()

# --- API Endpoint ---

@app.route("/process_image", methods=["POST"]) 
def handle_image_processing():
    
    file = request.files.get("image")
    algorithm = request.form.get("algorithm", "nn")
    k = int(request.form.get("k", 1))
    normType = request.form.get("normType", "cos")
    
    if not file:
        return jsonify({"error": "No image file provided"}), 400

    try:
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image_array_uint8 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image_array_uint8 is None:
            return jsonify({"error": "Failed to decode image"}), 400

        image_resized = cv2.resize(image_array_uint8, (92, 112))
        image_array_float = image_resized.flatten().astype(np.float32)

        results_data = get_recognition_results(
            image_array_float, algorithm, k, normType
        )
        
        return jsonify(results_data)

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred", "message": str(e)}), 500

@app.route('/preview', methods=['POST'])
def preview_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    
    try:
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image_array_uint8 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image_array_uint8 is None:
            return jsonify({"error": "Failed to decode PGM image"}), 400

        # resize it so the preview looks right
        image_resized = cv2.resize(image_array_uint8, (TARGET_SHAPE_2D[1], TARGET_SHAPE_2D[0]))

        # encode the resized 2D array as a PNG in memory
        is_success, buffer = cv2.imencode(".png", image_resized)
        
        # convert that in-memory PNG to a Base64 text string
        b64_string = base64.b64encode(buffer).decode("utf-8")

        # send the PNG string back to the frontend
        return jsonify({"image_b64": b64_string})

    except Exception as e:
        print(f"Preview error: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

@app.route("/run_statistics", methods=["POST"])
def run_statistics():
    """
    Runs a full benchmark on all 80 test images against all
    algorithm combinations and returns the accuracy and time.
    """
    print("--- Starting Statistics Benchmark ---")
    
    algorithms = ["nn", "knn"]
    k_values = [1, 3, 5, 7, 9]
    norm_types = ["cos", "2", "1", "inf"]
    
    benchmark_results = []
    
    for algo in algorithms:
        for norm in norm_types:
            if algo == 'nn':
                test_k_values = [1]
            else:
                test_k_values = k_values
                
            for k in test_k_values:
                if algo == 'nn' and k == 1 and any(r['name'] == f'K-NN (k=1) - {norm}' for r in benchmark_results):
                    continue

                total_correct = 0
        
                start_time = time.perf_counter()
                
                for i, test_img in enumerate(test_images):
                    true_label = (i // 2) + 1
                    
                    results = get_recognition_results(test_img, algo, k, norm)
                    
                    if results["person_label"] == true_label:
                        total_correct += 1
                        
                end_time = time.perf_counter()
                
                total_time_ms = (end_time - start_time) * 1000
                accuracy = (total_correct / len(test_images)) * 100
                
                if algo == 'nn':
                    name = f"NN - {norm}"
                else:
                    name = f"K-NN (k={k}) - {norm}"
                    
                benchmark_results.append({
                    "name": name,
                    "accuracy": accuracy,
                    "time_ms": total_time_ms
                })
                
                print(f"Test: {name} | Accuracy: {accuracy:.2f}% | Time: {total_time_ms:.2f} ms")

    print("--- Statistics Benchmark Finished ---")
    return jsonify(benchmark_results)

# --- Run the Server ---
if __name__ == "__main__":
    # Starts the web server on http://localhost:5000
    # The debug=True flag automatically reloads the server when you save changes.
    print("Starting Python backend server at http://localhost:5000")
    app.run(debug=True, port=5000)
