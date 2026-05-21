import os
import torch
import numpy as np
from PIL import Image
import faiss
import json
from transformers import CLIPProcessor, CLIPModel

class VisualSearchEngine:
    def __init__(self, index_folder, model_path=None):
        self.index_folder = index_folder
        self.model_path = model_path

        self.index_path = os.path.join(index_folder, "image_index.faiss")
        self.embeddings_path = os.path.join(index_folder, "embeddings.npy")
        self.paths_path = os.path.join(index_folder, "image_paths.json")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found at {self.index_path}. Run build_index.py first.")

        if not os.path.exists(self.paths_path):
            raise FileNotFoundError(f"Image paths file not found at {self.paths_path}. Run build_index.py first.")

        print("Loading CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        if self.model_path and os.path.exists(self.model_path):
            print(f"Loading fine-tuned model from {self.model_path}")
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        print("Loading FAISS index...")
        self.index = faiss.read_index(self.index_path)

        try:
            with open(self.paths_path, "r") as f:
                self.image_paths = json.load(f)
            print(f"Visual search engine initialized with {len(self.image_paths)} images")
        except json.JSONDecodeError:
            print("JSON format error, trying to load as text file...")
            with open(self.paths_path, "r") as f:
                self.image_paths = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(self.image_paths)} image paths from text file")

    def _normalize(self, x):
        return x / x.norm(dim=-1, keepdim=True)

    def encode_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                image_features = self._normalize(image_features)
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return image_features.cpu().numpy().astype("float32")
        except Exception as e:
            raise Exception(f"Error encoding image: {str(e)}")

    def encode_text(self, text):
        try:
            with torch.no_grad():
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)

                text_outputs = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )

                text_features = text_outputs.pooler_output
                text_features = self.model.text_projection(text_features)
                text_features = self._normalize(text_features)

            if self.device == "cuda":
                torch.cuda.empty_cache()
            return text_features.cpu().numpy().astype("float32")
        except Exception as e:
            raise Exception(f"Error encoding text: {str(e)}")

    def search_by_image(self, image_path, k=5):
        try:
            query_features = self.encode_image(image_path)
            distances, indices = self.index.search(query_features, k)

            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = distances[0][i]
                if 0 <= idx < len(self.image_paths):
                    results.append({
                        "image_path": self.image_paths[idx],
                        "similarity": float(score)
                    })
            return results
        except Exception as e:
            raise Exception(f"Error in image search: {str(e)}")

    def search_by_text(self, text, k=5):
        try:
            query_features = self.encode_text(text)
            distances, indices = self.index.search(query_features, k)

            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = distances[0][i]
                if 0 <= idx < len(self.image_paths):
                    results.append({
                        "image_path": self.image_paths[idx],
                        "similarity": float(score)
                    })
            return results
        except Exception as e:
            raise Exception(f"Error in text search: {str(e)}")