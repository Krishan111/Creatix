"""
Team Name: Creatix
Team Members: Siddharth Malkania, Krishan Verma , Rishi Mehrotra
Leaderboard Rank: 69

"""
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

class Preprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.98)
        self.img_size = 128

    def convert_to_jpg(self, input_path, output_path):
        try:
            with Image.open(input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=90)
                return True
        except Exception as e:
            print(f"Error converting {input_path}: {e}")
            return False

    def convert_all_images(self, source_dir, target_dir):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        all_files = os.listdir(source_dir)
        converted_files = {}
        
        for filename in tqdm(all_files, desc="Converting images"):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif')):
                source_path = os.path.join(source_dir, filename)
                name_without_ext = os.path.splitext(filename)[0]
                jpg_path = os.path.join(target_dir, f"{name_without_ext}.jpg")
                
                if self.convert_to_jpg(source_path, jpg_path):
                    converted_files[filename] = f"{name_without_ext}.jpg"
        return converted_files

    def extract_features(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Feature extraction logic (same as original)
            features = []
            # ... [original feature extraction code] ...
            
            return np.nan_to_num(features, nan=0.5, posinf=10.0, neginf=0.1)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def prepare_data(self, features):
        scaled = self.scaler.fit_transform(features)
        pca_features = self.pca.fit_transform(scaled)
        return pca_features