import numpy as np
import onnxruntime as ort
from PIL import Image

class FlowerClassifier:
    def __init__(self, onnx_path):
        # Explicitly set execution providers for CPU inference
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        # Broadcast-ready ImageNet statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def preprocess(self, image_path):
        img = Image.open(image_path).convert("RGB").resize((232, 232))
        # Center crop to 224x224
        left = (232 - 224) // 2
        top = (232 - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        
        # Scale, transpose channels to CHW, and add batch dimension
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return ((x - self.mean) / self.std).astype(np.float32)

    def predict(self, image_path, k=3):
        x = self.preprocess(image_path)
        logits = self.session.run(None, {"input": x})[0][0]
        
        # Stable Softmax operation
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        
        top = np.argsort(probs)[::-1][:k]
        return [(int(i), float(probs[i])) for i in top]