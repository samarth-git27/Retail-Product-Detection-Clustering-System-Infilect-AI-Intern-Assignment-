from typing import List
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from sklearn.cluster import DBSCAN

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Grouper:
    def __init__(self, device: str = DEVICE):
        self.device = device
        # Load ResNet50 backbone (pretrained). Remove final fc to get embeddings.
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        resnet.eval()
        resnet.to(self.device)
        self.model = resnet

        # transform
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def embedding_from_pil(self, pil_img: Image.Image):
        """Return L2-normalized embedding (1D numpy array)."""
        img_t = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img_t)  # (1, 2048)
        feat = feat.cpu().numpy().reshape(-1)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def embeddings_from_pil_list(self, pil_list: List[Image.Image]):
        """Return numpy array (N x D) of embeddings."""
        embs = []
        for img in pil_list:
            embs.append(self.embedding_from_pil(img))
        return np.vstack(embs)

    def cluster_embeddings(self, embeddings: np.ndarray, eps: float = 0.25, min_samples: int = 1):
        """
        Cluster using DBSCAN with cosine metric.
        eps: cosine distance threshold (0-1). Lower => stricter grouping.
        Returns labels array (N,)
        """
        if embeddings is None or len(embeddings) == 0:
            return np.array([], dtype=int)
        if embeddings.shape[0] == 1:
            return np.array([0], dtype=int)
        # DBSCAN with cosine metric
        cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = cluster.fit_predict(embeddings)
        return labels
