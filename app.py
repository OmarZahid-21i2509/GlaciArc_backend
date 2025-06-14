import os
import torch
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import requests
from scipy.ndimage import distance_transform_edt
from flask_cors import CORS
import subprocess
import ee
from zones import get_zones
import urllib.request




app = Flask(__name__)
CORS(app)  # Allow requests from Flutter Web


# ✅ Define folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load Pretrained U-Net Model
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=8,
    decoder_use_batchnorm=True
).to(device)

# ✅ Load Model Weights
MODEL_PATH = "ModelCheckpoint1.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1_Zj4jix2qAx6V6a-g0lSWfTULfGc7w42"  # ← your file's ID here
# ✅ Download if missing
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading Model.pth from Google Drive...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded.")

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")

# ✅ Define Preprocessing (Resize to 256×256)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ✅ Define Color Mapping
COLOR_MAP = {
    0: (255, 255, 0),  # Background
    1: (199, 252, 0),  # Glacier
    2: (254, 0, 86),   # House_Building
    3: (0, 183, 235),  # Ice
    4: (134, 34, 255), # Land
    5: (255, 255, 0),  # Undefined
    6: (0, 255, 206),  # Vegetation
    7: (255, 128, 0),  # Water
    "water": (255, 128, 0),   # Water (Orange) - Adjust if needed
    "vegetation": (0, 255, 206),  # Vegetation (Light Green)
    "red_zone": (255, 0, 0),   # Danger Zone (Red)
    "green_zone": (0, 100, 0)  # Safe Zone (Dark Green)
}


# ✅ Color Map (Updated from Notebook)
color_map = {
    0: (255, 255, 0),  # Background
    1: (199, 252, 0),  # Glacier
    2: (255, 0, 0),   # House_Building
    3: (0, 183, 235),  # Ice
    4: (134, 34, 255), # Land
    5: (255, 255, 0),  # Undefined
    6: (0, 255, 206),  # Vegetation
    7: (255, 128, 0),  # Water
    "water": (255, 128, 0),   # Water (Orange) - Adjust if needed
    "vegetation": (0, 255, 206),  # Vegetation (Light Green)
    "red_zone": (255, 0, 0),   # Danger Zone (Red)
    "green_zone": (0, 100, 0)  # Safe Zone (Dark Green)
}

# 🔹 API for Uploading Two Images
@app.route('/upload', methods=['POST'])
def upload():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Two images must be uploaded"}), 400

    image1 = request.files['image1']
    image2 = request.files['image2']

    filenames = [secure_filename(image1.filename), secure_filename(image2.filename)]
    paths = [os.path.join(app.config['UPLOAD_FOLDER'], filenames[0]), os.path.join(app.config['UPLOAD_FOLDER'], filenames[1])]

    try:
        image1.save(paths[0])
        image2.save(paths[1])

        resized_paths = []
        for i in range(2):
            img = Image.open(paths[i]).convert('RGB')
            resized_img = img.resize((256, 256))
            resized_filename = f"resized_{filenames[i]}"
            resized_path = os.path.join(app.config['OUTPUT_FOLDER'], resized_filename)

            # ✅ Ensure the resized image is saved correctly
            resized_img.save(resized_path)
            resized_paths.append(resized_filename)

            print(f"✅ Resized image saved at: {resized_path}")

        return jsonify({
            "original_image_urls": [
                f"http://localhost:5000/get_image/{filenames[0]}",
                f"http://localhost:5000/get_image/{filenames[1]}"
            ],
            "resized_image_urls": [
                f"http://localhost:5000/get_image/{resized_paths[0]}",
                f"http://localhost:5000/get_image/{resized_paths[1]}"
            ]
        })
    
    except Exception as e:
        return jsonify({"error": f"Failed to process images: {str(e)}"}), 500

# 🔹 API for Segmentation of Two Images
@app.route('/segment', methods=['POST'])
def segment():
    try:
        files = sorted(os.listdir(app.config['OUTPUT_FOLDER']))
        resized_files = [f for f in files if f.startswith("resized_")][-2:]

        if len(resized_files) < 2:
            return jsonify({"error": "Not enough images for segmentation"}), 404

        segmented_paths = []
        for file in resized_files:
            image_path = os.path.join(app.config['OUTPUT_FOLDER'], file)

            # ✅ Debugging: Check if file exists
            if not os.path.exists(image_path):
                return jsonify({"error": f"Resized image not found: {image_path}"}), 500

            print(f"Processing: {image_path}")

            # ✅ Load image and preprocess
            img = Image.open(image_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                output_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

            # ✅ Apply color mapping
            color_mapped_mask = np.zeros((output_mask.shape[0], output_mask.shape[1], 3), dtype=np.uint8)
            for class_index, color in COLOR_MAP.items():
                color_mapped_mask[output_mask == class_index] = color  

            # ✅ Save segmented image
            segmented_filename = f"segmented_{file}"
            segmented_path = os.path.join(app.config['OUTPUT_FOLDER'], segmented_filename)
            cv2.imwrite(segmented_path, cv2.cvtColor(color_mapped_mask, cv2.COLOR_RGB2BGR))
            segmented_paths.append(segmented_filename)

            if file == resized_files[-1]:
                overlay_path = os.path.join('static', 'segmented_overlay.png')

                # Convert from RGB to BGR (OpenCV default) before adding alpha
                bgr_image = cv2.cvtColor(color_mapped_mask, cv2.COLOR_RGB2BGR)
                alpha = np.ones(bgr_image.shape[:2], dtype=np.uint8) * 255
                bgra_image = cv2.merge((bgr_image, alpha))

                cv2.imwrite(overlay_path, bgra_image)
                print(f"✅ Overlay image saved for map: {overlay_path}")



            print(f"✅ Segmented image saved at: {segmented_path}")

        return jsonify({
            "resized_image_urls": [
                f"http://localhost:5000/get_image/{resized_files[0]}",
                f"http://localhost:5000/get_image/{resized_files[1]}"
            ],
            "segmented_image_urls": [
                f"http://localhost:5000/get_image/{segmented_paths[0]}",
                f"http://localhost:5000/get_image/{segmented_paths[1]}"
            ]
        })

    except Exception as e:
        return jsonify({"error": f"Segmentation failed: {str(e)}"}), 500


# 🔹 API for Comparing Two Segmented Images
@app.route('/compare', methods=['POST'])
def compare():
    try:
        files = sorted(os.listdir(app.config['OUTPUT_FOLDER']))
        segmented_files = [f for f in files if f.startswith("segmented_")][-2:]

        if len(segmented_files) < 2:
            return jsonify({"error": "Not enough segmented images for comparison"}), 404

        img1_path = os.path.join(app.config['OUTPUT_FOLDER'], segmented_files[0])  # First uploaded image
        img2_path = os.path.join(app.config['OUTPUT_FOLDER'], segmented_files[1])  # Second uploaded image

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            return jsonify({"error": "Segmented images not found"}), 500

        # ✅ Load images and convert to RGB (OpenCV loads in BGR)
        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

        if img1.shape != img2.shape:
            return jsonify({"error": "Segmented images must have the same dimensions for comparison"}), 400

        # ✅ Define New Color Mapping for Changes
        comparison_colors = {
            "vegetation_increase": (0, 100, 0),  # Dark Green
            "vegetation_decrease": (144, 238, 144),  # Light Green
            "water_increase": (0, 0, 139),  # Dark Blue
            "water_decrease": (173, 216, 230),  # Light Blue
            "glacier_increase": (204, 153, 0),  # Mustard
            "glacier_decrease": (255, 255, 153)  # Pale Yellow
        }

        # ✅ Define class RGB values from the segmentation process
        vegetation_color = np.array([0, 255, 206])  # Vegetation (Light Green)
        water_color = np.array([255, 128, 0])  # Water (Orange)
        glacier_color = np.array([199, 252, 0])  # Glacier (Light Green)

        # ✅ Define Tolerance for Color Matching (Fixes slight color compression issues)
        color_tolerance = 30  # Adjust based on variations in segmentation output

        # ✅ Create masks for each category in both images
        vegetation_mask_1 = np.all(np.isclose(img1, vegetation_color, atol=color_tolerance), axis=-1)
        vegetation_mask_2 = np.all(np.isclose(img2, vegetation_color, atol=color_tolerance), axis=-1)

        water_mask_1 = np.all(np.isclose(img1, water_color, atol=color_tolerance), axis=-1)
        water_mask_2 = np.all(np.isclose(img2, water_color, atol=color_tolerance), axis=-1)

        glacier_mask_1 = np.all(np.isclose(img1, glacier_color, atol=color_tolerance), axis=-1)
        glacier_mask_2 = np.all(np.isclose(img2, glacier_color, atol=color_tolerance), axis=-1)

        # ✅ Create an empty image for visualization
        diff_img = np.zeros_like(img1, dtype=np.uint8)

        # ✅ Apply color mapping based on increase/decrease conditions
        diff_img[vegetation_mask_2 & ~vegetation_mask_1] = comparison_colors["vegetation_increase"]  # Vegetation increased
        diff_img[~vegetation_mask_2 & vegetation_mask_1] = comparison_colors["vegetation_decrease"]  # Vegetation decreased

        diff_img[water_mask_2 & ~water_mask_1] = comparison_colors["water_increase"]  # Water increased
        diff_img[~water_mask_2 & water_mask_1] = comparison_colors["water_decrease"]  # Water decreased

        diff_img[glacier_mask_2 & ~glacier_mask_1] = comparison_colors["glacier_increase"]  # Glacier increased
        diff_img[~glacier_mask_2 & glacier_mask_1] = comparison_colors["glacier_decrease"]  # Glacier decreased

        # ✅ Save the new comparison result
        comparison_filename = "custom_comparison_result.png"
        comparison_path = os.path.join(app.config['OUTPUT_FOLDER'], comparison_filename)

        cv2.imwrite(comparison_path, cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR))  # Convert back to BGR for OpenCV

        print(f"✅ Custom Comparison Image Saved at: {comparison_path}")

        return jsonify({
            "comparison_image_url": f"http://localhost:5000/get_comparison_image/{comparison_filename}"
        })

    except Exception as e:
        return jsonify({"error": f"Comparison failed: {str(e)}"}), 500

# ✅ Serve the comparison image
@app.route('/get_comparison_image/<filename>', methods=['GET'])
def get_comparison_image(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return jsonify({"error": "Comparison file not found"}), 404


@app.route('/get_image/<filename>', methods=['GET'])
def get_image(filename):
    # Check both UPLOAD and OUTPUT folders for the image
    file_path = os.path.join(OUTPUT_FOLDER, filename)  # First check OUTPUT folder
    if not os.path.exists(file_path):  
        file_path = os.path.join(UPLOAD_FOLDER, filename)  # Then check UPLOAD folder

    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/jpeg')

    return jsonify({"error": "File not found"}), 404



# ✅ Function to Apply Colors
def apply_custom_colors(mask, color_map):
    """ Converts class mask to RGB """
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_value, color in color_map.items():
        rgb_mask[mask == class_value] = color
    return rgb_mask

# ✅ Function to Apply Red-Green Zoning on Existing Colors
def apply_red_green_zoning(image, radius=15):
    """ Detects vegetation & water from the segmented image and applies red-green zoning """
    image_array = np.array(image)

    # Convert predefined colors to NumPy arrays for comparison
    water_color = np.array(COLOR_MAP["water"])
    vegetation_color = np.array(COLOR_MAP["vegetation"])

    # Tolerance for color comparison
    color_tolerance = 30  # Adjust based on variations in segmentation output

    # Create binary masks using np.isclose to allow some variation in color
    water_mask = np.all(np.isclose(image_array, water_color, atol=color_tolerance), axis=-1)
    vegetation_mask = np.all(np.isclose(image_array, vegetation_color, atol=color_tolerance), axis=-1)

    # Compute Distance Transform (distance from each pixel to nearest water pixel)
    distance_to_water = distance_transform_edt(~water_mask)

    # Apply Red-Green Logic:
    vegetation_near_water = (distance_to_water <= radius) & vegetation_mask
    vegetation_far_from_water = (distance_to_water > radius) & vegetation_mask

    # Make a copy of the original image
    processed_image = image_array.copy()

    # Apply zoning colors
    processed_image[vegetation_near_water] = COLOR_MAP["red_zone"]  # Red for vegetation near water
    processed_image[vegetation_far_from_water] = COLOR_MAP["green_zone"]  # Dark Green for far vegetation

    return processed_image

# ✅ Flask API for Analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image_url' not in request.form:
        return jsonify({"error": "No image URL provided"}), 400

    image_url = request.form['image_url']
    print(f"📥 Received image URL: {image_url}")

    # ✅ Download Image
    upload_path = os.path.join(UPLOAD_FOLDER, "analysis_image.jpg")
    output_path = os.path.join(OUTPUT_FOLDER, "processed_analysis_image.jpg")

    response = requests.get(image_url)
    if response.status_code == 200:
        with open(upload_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Image downloaded at {upload_path}")
    else:
        return jsonify({"error": "Failed to download image"}), 500

    # ✅ Load Image in RGB Mode
    image = Image.open(upload_path).convert("RGB")

    # ✅ Apply Red-Green Zoning
    processed_image = apply_red_green_zoning(image)

    # ✅ Save Processed Image
    cv2.imwrite(output_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
    print(f"✅ Processed image saved at {output_path}")

    # ✅ Return Processed Image URL
    processed_image_url = f"http://localhost:5000/get_image/{os.path.basename(output_path)}"
    return jsonify({"message": "Analysis completed", "processed_image_url": processed_image_url})

@app.route('/results', methods=['GET'])
def get_results():
    try:
        # Get the latest files from the respective folders
        files = sorted(os.listdir(OUTPUT_FOLDER))

        # Find original images
        original_files = [f for f in os.listdir(UPLOAD_FOLDER) if not f.startswith("resized_")][-2:]
        original_urls = [
            f"http://localhost:5000/get_image/{original_files[0]}" if len(original_files) > 0 else None,
            f"http://localhost:5000/get_image/{original_files[1]}" if len(original_files) > 1 else None,
        ]

        # Find resized images
        resized_files = [f for f in files if f.startswith("resized_")][-2:]
        resized_urls = [
            f"http://localhost:5000/get_image/{resized_files[0]}" if len(resized_files) > 0 else None,
            f"http://localhost:5000/get_image/{resized_files[1]}" if len(resized_files) > 1 else None,
        ]

        # Find segmented images
        segmented_files = [f for f in files if f.startswith("segmented_")][-2:]
        segmented_urls = [
            f"http://localhost:5000/get_image/{segmented_files[0]}" if len(segmented_files) > 0 else None,
            f"http://localhost:5000/get_image/{segmented_files[1]}" if len(segmented_files) > 1 else None,
        ]

        # Find comparison image
        comparison_files = [f for f in files if f.startswith("custom_comparison_result")]
        comparison_url = f"http://localhost:5000/get_image/{comparison_files[0]}" if comparison_files else None

        # Find analysis image
        analysis_files = [f for f in files if f.startswith("processed_analysis_image")]
        analysis_url = f"http://localhost:5000/get_image/{analysis_files[0]}" if analysis_files else None

        # Validate that at least the essential images exist
        if not all([original_urls[0], resized_urls[0], segmented_urls[0]]):
            return jsonify({"error": "Missing required images"}), 404

        return jsonify({
            "original_image_urls": original_urls,
            "resized_image_urls": resized_urls,
            "segmented_image_urls": segmented_urls,
            "comparison_image_url": comparison_url,
            "analysis_image_url": analysis_url
        })

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve results: {str(e)}"}), 500


@app.route('/get_glacier_map', methods=['GET'])
def get_glacier_map():
    glacier = request.args.get('glacier')
    year = request.args.get('year')

    if not glacier or not year:
        return jsonify({"error": "Missing glacier or year parameter"}), 400

    map_filename = f"{glacier}_{year}_map.html"
    map_path = os.path.join("static", map_filename)

    # ✅ Run glacier.py to generate the map if missing
    if not os.path.exists(map_path):
        try:
            print(f"Generating map for {glacier} ({year})...")
            subprocess.run(["python", "glacier.py", glacier, year], check=True)

            if not os.path.exists(map_path):
                return jsonify({"error": "Failed to generate map"}), 500

        except subprocess.CalledProcessError as e:
            return jsonify({"error": str(e)}), 500

    # ✅ Serve the file
    print(f"✅ Serving {map_filename}...")
    return send_from_directory("static", map_filename)





@app.route('/fetch_image', methods=['POST'])
def fetch_image():
    try:
        ee.Initialize(project='glaciarc-project')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='glaciarc-project')

    ZONES = get_zones()
    data = request.json

    glacier = data['glacier']
    zone = data['zone']
    year1 = data['year1']
    month1 = data['month1']
    year2 = data['year2']
    month2 = data['month2']
    cloud = data['cloud']

    polygon = ZONES[glacier][zone].buffer(1000)  # Increased buffer for larger area

    def get_image(year, month):
        start_date = f"{year}-{month.zfill(2)}-01"
        end_date = f"{year}-{month.zfill(2)}-28"  # Safe upper bound for February
        image = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
            .filterBounds(polygon) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud)) \
            .select(['B4', 'B3', 'B2']) \
            .median()
        return image

    try:
        image1 = get_image(year1, month1)
        image2 = get_image(year2, month2)

        url1 = image1.getThumbURL({
            'region': polygon,
            'dimensions': 2048,
            'format': 'jpg',
            'min': 300,
            'max': 2500,
            'gamma': 1.2
        })

        url2 = image2.getThumbURL({
            'region': polygon,
            'dimensions': 2048,
            'format': 'jpg',
            'min': 300,
            'max': 2500,
            'gamma': 1.2
        })

        return jsonify({
            "success": True,
            "url1": url1,
            "url2": url2
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/segmentation_map')
def segmentation_map():
    return send_file('templates/segmentation_map.html')

    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)