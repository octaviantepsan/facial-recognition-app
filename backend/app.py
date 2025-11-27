import numpy as np
import cv2
import os
import base64
import time
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy import stats

# NO sklearn. Pure Numpy.

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, allow_headers=["Content-Type"])

# --- Global Variables ---
train_images = []
test_images = []

# --- MATH GLOBALS ---
mean_face = None 
eigenfaces_standard_full = None
eigenfaces_svd_full = None
eigenfaces_lanczos_full = None 

# Data Counts
images_per_person_train = 0 
images_per_person_test = 0  

TARGET_SHAPE_2D = (112, 92) 
TARGET_SHAPE_CV2 = (92, 112) 

def calculate_class_means(weights):
    if weights is None or len(weights) == 0: return None
    means_list = []
    total_train = weights.shape[0]
    
    if images_per_person_train > 0:
        for i in range(0, total_train, images_per_person_train):
            person_vectors = weights[i : i + images_per_person_train]
            if len(person_vectors) > 0:
                person_mean = np.mean(person_vectors, axis=0)
                means_list.append(person_mean)
                
    return np.array(means_list)

def train_pca_standard():
    global mean_face, eigenfaces_standard_full
    if len(train_images) == 0: return
    print("Training PCA (standard)")
    if mean_face is None: mean_face = np.mean(train_images, axis=0)
    
    A = train_images - mean_face
    L = np.dot(A, A.T)
    
    eigvals, eigvecs_small = np.linalg.eigh(L)
    idx = np.argsort(eigvals)[::-1]
    eigvecs_small = eigvecs_small[:, idx]
    eigenfaces = np.dot(A.T, eigvecs_small)
    
    for i in range(eigenfaces.shape[1]):
        norm = np.linalg.norm(eigenfaces[:, i])
        if norm > 1e-10: eigenfaces[:, i] /= norm
        
    eigenfaces_standard_full = eigenfaces.T

def train_pca_svd():
    global mean_face, eigenfaces_svd_full
    if len(train_images) == 0: return
    print("Training PCA (SVD)")
    
    if mean_face is None: mean_face = np.mean(train_images, axis=0)
    
    A = train_images - mean_face
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    eigenfaces_svd_full = Vt 

def lanczos_algorithm_scratch(A, k):
    n_samples, n_features = A.shape
    m = k + 2 
    if m > n_features: m = n_features
    V = np.zeros((m, n_features))
    alphas = np.zeros(m); betas = np.zeros(m - 1)
    
    v = np.random.rand(n_features); v = v / np.linalg.norm(v)
    V[0, :] = v
    def implicit_multiply(vec): return np.dot(A.T, np.dot(A, vec))
    w = implicit_multiply(v); alpha = np.dot(w, v); alphas[0] = alpha; w = w - alpha * v
    
    for j in range(1, m):
        beta = np.linalg.norm(w)
        if beta < 1e-10: break
        betas[j-1] = beta; v_next = w / beta; V[j, :] = v_next
        w = implicit_multiply(v_next); w = w - beta * V[j-1, :]; alpha = np.dot(w, v_next); alphas[j] = alpha; w = w - alpha * v_next
    
    T = np.diag(alphas[:j+1]) + np.diag(betas[:j], 1) + np.diag(betas[:j], -1)
    _, eigvecs_T = np.linalg.eigh(T)
    final_eigvecs = np.dot(V[:j+1, :].T, eigvecs_T)
    
    return final_eigvecs[:, -k:][:, ::-1]

def train_lanczos_scratch():
    global eigenfaces_lanczos_full, mean_face
    
    if len(train_images) == 0: return
    print("Training Lanczos")
    print()
    
    
    if mean_face is None: mean_face = np.mean(train_images, axis=0)
    
    A = train_images - mean_face
    k_max = min(150, len(train_images) - 1)
    eigenfaces = lanczos_algorithm_scratch(A, k_max)
    
    for i in range(eigenfaces.shape[1]):
        norm = np.linalg.norm(eigenfaces[:, i])
        if norm > 1e-10: eigenfaces[:, i] /= norm
        
    eigenfaces_lanczos_full = eigenfaces.T

def get_sliced_model(feature_mode, n_components):
    if n_components == 'max': k = 99999
    else: k = int(n_components)

    full_model = None
    if feature_mode == 'eigen_snapshot' or feature_mode == 'eigen_mean_snapshot':
        full_model = eigenfaces_standard_full
    elif feature_mode == 'eigen_svd' or feature_mode == 'eigen_mean_svd':
        full_model = eigenfaces_svd_full
    elif feature_mode == 'lanczos':
        full_model = eigenfaces_lanczos_full
    
    if full_model is None: return None, None
    
    limit = min(k, full_model.shape[0])
    sliced_eigenfaces = full_model[:limit]
    A = train_images - mean_face
    train_weights_sliced = np.dot(A, sliced_eigenfaces.T)
    
    return sliced_eigenfaces, train_weights_sliced

def reconstruct_image(weights, eigenfaces):
    reconstruction = np.dot(weights, eigenfaces) + mean_face
    reconstruction = reconstruction.reshape(TARGET_SHAPE_2D)
    
    norm_image = cv2.normalize(reconstruction, None, 0, 255, cv2.NORM_MINMAX)
    norm_image = norm_image.astype(np.uint8)
    
    is_success, buffer = cv2.imencode(".png", norm_image)
    return base64.b64encode(buffer).decode("utf-8")

def load_data(dataset_name="attfaces", train_ratio=0.8):
    global train_images, test_images, images_per_person_train, images_per_person_test, mean_face
    print()
    print(f"Loading Dataset: {dataset_name}")
    mean_face = None
    train_images = []
    
    base_path_laptop = rf"C:\Octavian\github\facial-recognition\facial-recognition-app\assets\{dataset_name}"
    base_path_pc = rf"D:\OCTAVIAN\github\facial-recognition-app\assets\{dataset_name}"
    base_path_to_use = os.path.normpath(base_path_pc)
    if not os.path.exists(base_path_to_use):
        base_path_to_use = os.path.normpath(base_path_laptop)
        if not os.path.exists(base_path_to_use): return False

    try:
        items = os.listdir(base_path_to_use)
        dir_list = [d for d in items if os.path.isdir(os.path.join(base_path_to_use, d))]
        dir_list = sorted(dir_list, key=lambda x: int(x[1:]) if x.startswith('s') else int(x) if x.isdigit() else x)
    except: dir_list = sorted(os.listdir(base_path_to_use))

    total_imgs_per_class = 10 
    if len(dir_list) > 0:
        first_path = os.path.join(base_path_to_use, dir_list[0])
        valid_imgs = [f for f in os.listdir(first_path) if f.lower().endswith(('.pgm', '.jpg', '.png'))]
        if len(valid_imgs) > 0: total_imgs_per_class = len(valid_imgs)
    
    split_count = math.ceil(total_imgs_per_class * train_ratio)
    if split_count == total_imgs_per_class and total_imgs_per_class > 1: split_count -= 1
    
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
        except: images = sorted(os.listdir(class_path))
            
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
    
    if len(train_images) == 0: return False

    train_pca_standard()
    train_pca_svd()
    train_lanczos_scratch()
    return True

def calcDist(test_sample, database, normType='2'):
    if normType == 'cos':
        dot_products = np.dot(database, test_sample)
        train_norms = np.linalg.norm(database, axis=1)
        test_norm = np.linalg.norm(test_sample)
        epsilon = 1e-10
        cosine_similarity = dot_products / ((train_norms * test_norm) + epsilon)
        return 1 - cosine_similarity
    
    ord_val = 2
    if normType == '1': ord_val = 1
    elif normType == 'inf': ord_val = np.inf
    return np.linalg.norm(database - test_sample, ord=ord_val, axis=1)

def prepareImgToSend(matched_img_index):
    matched_img = train_images[matched_img_index]
    matched_img_2d = matched_img.reshape(TARGET_SHAPE_2D)
    matched_image_uint8 = matched_img_2d.astype(np.uint8)
    _, buf = cv2.imencode(".png", matched_image_uint8)
    return base64.b64encode(buf).decode("utf-8")

def run_nn_algorithm(test_vector, database, is_class_reps=False, normType='2'):
    distances = calcDist(test_vector, database, normType)
    return int(np.argmin(distances))

def run_knn_algorithm(test_vector, database, k=1, normType='2'):
    distances = calcDist(test_vector, database, normType)
    k_nearest_indices = np.argsort(distances)[:k]
    k_person_labels = [(idx // images_per_person_train) + 1 for idx in k_nearest_indices]
    most_common = stats.mode(k_person_labels)[0]
    if isinstance(most_common, np.ndarray): most_common = most_common[0]
    return {"person_label": int(most_common), "nearest_idx": int(k_nearest_indices[0])}

def get_recognition_results(image_array_float, feature_mode, algorithm, k, normType, n_components):
    if len(train_images) == 0: raise Exception("DB empty")
    ghost_image_b64 = None

    if feature_mode == 'raw':
        test_vector = image_array_float
        database = train_images
        is_class_reps = False
        algo_prefix = ""
    else:
        sliced_eigenfaces, sliced_train_weights = get_sliced_model(feature_mode, n_components)
        if sliced_eigenfaces is None: raise Exception("Model not trained")
        
        centered = image_array_float - mean_face
        test_vector = np.dot(sliced_eigenfaces, centered)
        
        if 'mean' in feature_mode:
            database = calculate_class_means(sliced_train_weights)
            is_class_reps = True
            algo_prefix = "Eigen(Mean) "
            algorithm = 'nn' 
        else:
            database = sliced_train_weights
            is_class_reps = False
            algo_prefix = "Eigen "
            
        if feature_mode == 'lanczos': algo_prefix = "Lanczos "
        ghost_image_b64 = reconstruct_image(test_vector, sliced_eigenfaces)

    if algorithm == 'nn':
        nearest_idx = run_nn_algorithm(test_vector, database, is_class_reps, normType=normType)
        if is_class_reps:
            person_label = nearest_idx + 1
            display_idx = (person_label - 1) * images_per_person_train
        else:
            person_label = (nearest_idx // images_per_person_train) + 1
            display_idx = nearest_idx
        return {
            "algorithm": f"{algo_prefix}NN",
            "person_label": person_label,
            "nearest_idx": display_idx,
            "image_b64": prepareImgToSend(display_idx),
            "ghost_b64": ghost_image_b64
        }
    else:
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

# --- 5. Endpoints ---

@app.route("/load_dataset", methods=["POST"])
def api_load_dataset():
    data = request.json
    success = load_data(data.get('dataset'), float(data.get('split')))
    if success: return jsonify({"message": "Loaded", "train_shape": train_images.shape})
    return jsonify({"error": "Path"}), 400

@app.route("/process_image", methods=["POST"]) 
def handle_image_processing():
    if "image" not in request.files: return jsonify({"error": "No image"}), 400
    file = request.files["image"]
    feature_mode = request.form.get("feature_mode", "raw")
    algo = request.form.get("algorithm", "nn")
    k = int(request.form.get("k", 1))
    norm = request.form.get("normType", "cos")
    n_components = request.form.get("n_components", 50)

    try:
        img_data = file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        img_array = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img_array is None: return jsonify({"error": "Bad image"}), 400
        img_resized = cv2.resize(img_array, TARGET_SHAPE_CV2)
        img_flat = img_resized.flatten().astype(np.float32)
        res = get_recognition_results(img_flat, feature_mode, algo, k, norm, n_components)
        return jsonify(res)
    except Exception as e:
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
        return jsonify({"image_b64": base64.b64encode(buf).decode("utf-8")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/run_statistics", methods=["POST"])
def run_statistics():
    print("--- Starting Statistics Benchmark ---")
    
    classification_results = []
    raw_algorithms = ["nn", "knn"]
    raw_k = [1, 3, 5, 7]
    raw_norms = ["cos", "2", "1", "inf"]
    
    for algo in raw_algorithms:
        k_list = [1] if algo == 'nn' else raw_k
        for norm in raw_norms:
            for k in k_list:
                correct = 0
                start = time.perf_counter()
                for i, test_img in enumerate(test_images):
                    true_label = (i // images_per_person_test) + 1
                    res = get_recognition_results(test_img, 'raw', algo, k, norm, 0)
                    if res["person_label"] == true_label: correct += 1
                dur = (time.perf_counter() - start) * 1000
                acc = (correct / len(test_images)) * 100
                
                name = f"Raw | {algo.upper()}({k}) | {norm}"
                classification_results.append({"name": name, "accuracy": acc, "time_ms": dur})
                print(f"{name} -> {acc:.1f}%")

    preprocessing_results = []
    pca_modes = ["eigen_snapshot", "eigen_svd", "lanczos"]
    component_counts = [30, 50, 80, 100]
    
    for mode in pca_modes:
        for comp in component_counts:
            
            start = time.perf_counter()
            sliced_ef, sliced_weights = get_sliced_model(mode, comp)
            dur = (time.perf_counter() - start) * 1000
            
            # accuracy uses simple NN-Cosine as benchmark
            correct = 0
            for i, test_img in enumerate(test_images):
                true_label = (i // images_per_person_test) + 1

                res = get_recognition_results(test_img, mode, 'nn', 1, 'cos', comp)
                if res["person_label"] == true_label: correct += 1
            acc = (correct / len(test_images)) * 100
            
            mode_name = "Snap" if mode == 'eigen_snapshot' else "SVD" if mode == 'eigen_svd' else "Lanczos"
            name = f"{mode_name} | Comp={comp}"
            preprocessing_results.append({"name": name, "accuracy": acc, "time_ms": dur})
            print(f"{name} -> {dur:.2f}ms")

    return jsonify({
        "classification": classification_results,
        "preprocessing": preprocessing_results
    })

if __name__ == "__main__":
    load_data("attfaces", 0.8)
    app.run(debug=True, port=5000)