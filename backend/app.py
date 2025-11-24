import numpy as np
import cv2
import os
import base64
import time
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy import stats
from sklearn.decomposition import PCA

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type"])

# --- Global Variables ---
train_images = []
test_images = []

# --- PCA GLOBALS ---
pca_model = None
train_weights = []        # For Standard Eigenfaces
train_weights_means = []  # For Class Representatives

# Data Counts (Updated dynamically)
images_per_person_train = 0 
images_per_person_test = 0  

TARGET_SHAPE_2D = (112, 92) 
TARGET_SHAPE_CV2 = (92, 112) 

# --- 1. Data Loading & Processing ---

def train_pca_model():
    """
    Trains PCA on the current train_images and prepares weights.
    """
    global pca_model, train_weights, train_weights_means
    
    if len(train_images) == 0: return

    print("--- Training PCA Model ---")
    
    # 1. Fit PCA (Keep 95% variance)
    pca_model = PCA(n_components=0.95)
    train_weights = pca_model.fit_transform(train_images)
    
    print(f"PCA Trained. Reduced features from {train_images.shape[1]} to {train_weights.shape[1]}")
    
    # 2. Calculate Class Representatives (Means)
    means_list = []
    total_train = train_weights.shape[0]
    
    if images_per_person_train > 0:
        for i in range(0, total_train, images_per_person_train):
            person_vectors = train_weights[i : i + images_per_person_train]
            person_mean = np.mean(person_vectors, axis=0)
            means_list.append(person_mean)
            
    train_weights_means = np.array(means_list)
    print(f"Class Representatives calculated. Shape: {train_weights_means.shape}")

def reconstruct_image_from_weights(weights):
    """
    Takes the PCA weights and rebuilds the image (The Ghost).
    """
    if pca_model is None: return None

    # 1. Inverse Transform: Weights -> Flattened Pixels
    # weights shape is (1, n_components)
    reconstruction = pca_model.inverse_transform(weights)
    
    # 2. Reshape back to 2D
    reconstruction = reconstruction.reshape(TARGET_SHAPE_2D)
    
    # 3. Normalize to 0-255 range for display
    norm_image = cv2.normalize(reconstruction, None, 0, 255, cv2.NORM_MINMAX)
    norm_image = norm_image.astype(np.uint8)
    
    # 4. Encode to Base64
    is_success, buffer = cv2.imencode(".png", norm_image)
    b64_string = base64.b64encode(buffer).decode("utf-8")
    
    return b64_string

def load_data(dataset_name="attfaces", train_ratio=0.8):
    global train_images, test_images, images_per_person_train, images_per_person_test
    
    print(f"--- Loading Dataset: {dataset_name} with split {train_ratio} ---")
    
    # Construct paths
    base_path_laptop = rf"C:\Octavian\github\facial-recognition\facial-recognition-app\assets\{dataset_name}"
    base_path_pc = rf"D:\OCTAVIAN\github\facial-recognition-app\assets\{dataset_name}"

    base_path_to_use = os.path.normpath(base_path_pc)
    if not os.path.exists(base_path_to_use):
        base_path_to_use = os.path.normpath(base_path_laptop)
        if not os.path.exists(base_path_to_use):
            print(f"ERROR: Path for {dataset_name} does not exist.")
            return False

    try:
        # Sort folders numerically (s1, s2...) or naturally
        items = os.listdir(base_path_to_use)
        dir_list = [d for d in items if os.path.isdir(os.path.join(base_path_to_use, d))]
        dir_list = sorted(dir_list, key=lambda x: int(x[1:]) if x.startswith('s') else int(x) if x.isdigit() else x)
    except:
        dir_list = sorted(os.listdir(base_path_to_use))

    # Dynamic counting of images per class
    total_imgs_per_class = 10 # Default fallback
    if len(dir_list) > 0:
        first_path = os.path.join(base_path_to_use, dir_list[0])
        valid_imgs = [f for f in os.listdir(first_path) if f.lower().endswith(('.pgm', '.jpg', '.png'))]
        if len(valid_imgs) > 0:
            total_imgs_per_class = len(valid_imgs)
    
    # Update Counts
    split_count = math.ceil(total_imgs_per_class * train_ratio)
    # Safety: if ratio is 1.0, leave at least 1 for testing if possible
    if split_count == total_imgs_per_class and total_imgs_per_class > 1:
        split_count = total_imgs_per_class - 1
    
    images_per_person_train = split_count
    images_per_person_test = total_imgs_per_class - split_count
    
    print(f"Config: {images_per_person_train} Train / {images_per_person_test} Test per person")

    train_images_list = []
    test_images_list = []

    for class_dir in dir_list:
        class_path = os.path.join(base_path_to_use, class_dir)
        try:
            all_files = os.listdir(class_path)
            images = sorted([f for f in all_files if f.lower().endswith(('.pgm', '.jpg', '.png'))], key=lambda x: int(x.split('.')[0]))
        except:
            images = sorted(os.listdir(class_path))
            
        train_imgs_names = images[:images_per_person_train]
        test_imgs_names = images[images_per_person_train:]

        for img_name in train_imgs_names:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != TARGET_SHAPE_2D: img = cv2.resize(img, TARGET_SHAPE_CV2)
                train_images_list.append(img.flatten().astype(np.float32))

        for img_name in test_imgs_names:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != TARGET_SHAPE_2D: img = cv2.resize(img, TARGET_SHAPE_CV2)
                test_images_list.append(img.flatten().astype(np.float32))

    train_images = np.array(train_images_list)
    test_images = np.array(test_images_list)

    print(f"Loaded: train_shape={train_images.shape}, test_shape={test_images.shape}")
    
    # Train PCA immediately
    train_pca_model()
    
    return True

# --- 2. Algorithms ---

def calcDist(test_sample, database, normType='2'):
    """
    Calculates distance between test_sample and a generic database.
    Database can be: Raw Pixels, Eigen Weights, or Mean Weights.
    """
    if normType == 'cos':
        dot_products = np.dot(database, test_sample)
        train_norms = np.linalg.norm(database, axis=1)
        test_norm = np.linalg.norm(test_sample)
        epsilon = 1e-10
        cosine_similarity = dot_products / ((train_norms * test_norm) + epsilon)
        return 1 - cosine_similarity
    
    if normType == '1': ord_val = 1
    elif normType == 'inf': ord_val = np.inf
    else: ord_val = 2
        
    distances = np.linalg.norm(database - test_sample, ord=ord_val, axis=1)
    return distances

def prepareImgToSend(matched_img_index):
    matched_img = train_images[matched_img_index]
    matched_img_2d = matched_img.reshape(TARGET_SHAPE_2D)
    matched_image_uint8 = matched_img_2d.astype(np.uint8)
    is_success, buffer = cv2.imencode(".png", matched_image_uint8)
    b64_string = base64.b64encode(buffer).decode("utf-8")
    return b64_string

def run_nn_algorithm(test_vector, database, is_class_reps=False, normType='2'):
    distances = calcDist(test_vector, database, normType)
    nearest_idx = int(np.argmin(distances))
    return nearest_idx

def run_knn_algorithm(test_vector, database, k=1, normType='2'):
    distances = calcDist(test_vector, database, normType)
    k_nearest_indices = np.argsort(distances)[:k]
    
    # Dynamic Label Calculation
    k_person_labels = [(idx // images_per_person_train) + 1 for idx in k_nearest_indices]
    
    most_common_label = stats.mode(k_person_labels)[0]
    if isinstance(most_common_label, np.ndarray): most_common_label = most_common_label[0]
    
    single_nearest_idx = int(k_nearest_indices[0])
    
    return {
        "person_label": int(most_common_label),
        "nearest_idx": single_nearest_idx
    }

def get_recognition_results(image_array_float, feature_mode, algorithm, k, normType):
    """
    Main logic router.
    """
    ghost_image_b64 = None

    # 1. DETERMINE DATA TO USE
    if feature_mode == 'raw':
        test_vector = image_array_float
        database = train_images
        is_class_reps = False
        algo_prefix = ""
        
    elif feature_mode == 'eigen':
        # Project
        weights_vector = pca_model.transform(image_array_float.reshape(1, -1))
        test_vector = weights_vector.flatten()
        database = train_weights
        is_class_reps = False
        algo_prefix = "Eigen "
        
        # Create Ghost
        ghost_image_b64 = reconstruct_image_from_weights(weights_vector)
        
    elif feature_mode == 'eigen_mean':
        # Project
        weights_vector = pca_model.transform(image_array_float.reshape(1, -1))
        test_vector = weights_vector.flatten()
        database = train_weights_means
        is_class_reps = True
        algo_prefix = "Eigen(Mean) "
        algorithm = 'nn' # Force NN

        # Create Ghost
        ghost_image_b64 = reconstruct_image_from_weights(weights_vector)

    # 2. RUN ALGORITHM
    if algorithm == 'nn':
        nearest_idx = run_nn_algorithm(test_vector, database, is_class_reps, normType=normType)
        
        if is_class_reps:
            # For Class Reps, index 0 is Person 1
            person_label = nearest_idx + 1
            display_idx = (person_label - 1) * images_per_person_train
        else:
            # Standard logic
            person_label = (nearest_idx // images_per_person_train) + 1
            display_idx = nearest_idx
        
        img_as_string = prepareImgToSend(display_idx)
        algo_name = f"{algo_prefix}NN"

        return {
            "algorithm": algo_name,
            "person_label": person_label,
            "nearest_idx": display_idx,
            "image_b64": img_as_string,
            "ghost_b64": ghost_image_b64
        }

    else: # K-NN
        results_dict = run_knn_algorithm(test_vector, database, k=k, normType=normType)
        person_label = results_dict["person_label"]
        nearest_idx = results_dict["nearest_idx"]
        
        if k == 1:
            img_as_string = prepareImgToSend(nearest_idx)
            algo_name = f"{algo_prefix}K-NN (k=1)"
        else:
            first_index_of_winner = (person_label - 1) * images_per_person_train
            img_as_string = prepareImgToSend(first_index_of_winner)
            algo_name = f"{algo_prefix}K-NN (k={k})"

        return {
            "algorithm": algo_name,
            "person_label": person_label,
            "nearest_idx": nearest_idx,
            "image_b64": img_as_string,
            "ghost_b64": ghost_image_b64
        }

# --- 3. API Endpoints ---

@app.route("/load_dataset", methods=["POST"])
def api_load_dataset():
    data = request.json
    dataset_name = data.get('dataset', 'attfaces')
    split_ratio = float(data.get('split', 0.8))
    success = load_data(dataset_name, split_ratio)
    if success:
        return jsonify({"message": f"Loaded {dataset_name}", "train_shape": train_images.shape})
    else:
        return jsonify({"error": "Dataset path not found"}), 400

@app.route("/process_image", methods=["POST"]) 
def handle_image_processing():
    if "image" not in request.files: return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    feature_mode = request.form.get("feature_mode", "raw")
    algo = request.form.get("algorithm", "nn")
    k = int(request.form.get("k", 1))
    norm = request.form.get("normType", "cos")

    try:
        img_data = file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_array is None: return jsonify({"error": "Bad image"}), 400

        img_resized = cv2.resize(img_array, TARGET_SHAPE_CV2)
        img_flat = img_resized.flatten().astype(np.float32)

        res = get_recognition_results(img_flat, feature_mode, algo, k, norm)
        return jsonify(res)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/preview', methods=['POST'])
def preview_image():
    if 'image' not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files['image']
    try:
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None: return jsonify({"error": "Decode fail"}), 400
        img_res = cv2.resize(img, TARGET_SHAPE_CV2)
        _, buf = cv2.imencode(".png", img_res)
        b64 = base64.b64encode(buf).decode("utf-8")
        return jsonify({"image_b64": b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/run_statistics", methods=["POST"])
def run_statistics():
    print("--- Starting Statistics Benchmark ---")
    
    feature_modes = ["raw", "eigen", "eigen_mean"]
    algorithms = ["nn", "knn"]
    k_values = [1, 3, 5, 7, 9]
    norm_types = ["cos", "2", "1", "inf"]
    
    benchmark_results = []
    
    for feat in feature_modes:
        for algo in algorithms:
            if feat == 'eigen_mean' and algo == 'knn': continue 
            
            test_k = [1] if algo == 'nn' else k_values
            for norm in norm_types:
                for k in test_k:
                    correct = 0
                    start = time.perf_counter()
                    
                    for i, test_img in enumerate(test_images):
                        # True label calculation using current test count
                        true_label = (i // images_per_person_test) + 1
                        res = get_recognition_results(test_img, feat, algo, k, norm)
                        if res["person_label"] == true_label:
                            correct += 1
                    
                    duration = (time.perf_counter() - start) * 1000
                    acc = (correct / len(test_images)) * 100
                    
                    algo_str = "NN" if algo == 'nn' else f"K-NN({k})"
                    feat_str = "Raw" if feat == 'raw' else "Eigen" if feat == 'eigen' else "EigenMean"
                    
                    name = f"{feat_str} | {algo_str} | {norm}"
                    benchmark_results.append({"name": name, "accuracy": acc, "time_ms": duration})
                    print(f"{name}: {acc:.2f}%")

    return jsonify(benchmark_results)

if __name__ == "__main__":
    # Load default
    load_data("attfaces", 0.8)
    app.run(debug=True, port=5000)