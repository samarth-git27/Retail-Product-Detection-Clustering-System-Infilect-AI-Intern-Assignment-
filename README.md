# Retail-Product-Detection-Clustering-System-Infilect-AI-Intern-Assignment-
This project implements a full computer vision pipeline for retail shelf analytics, including:  Object Detection (YOLOv8n)  Feature Embedding (ResNet50)  Similarity Clustering (DBSCAN)  Product Grouping  Visualization  Fully functioning REST API (Flask)
The system processes an input retail shelf image and returns:

✔ Detected products
✔ Embeddings for each product
✔ Cluster IDs
✔ Grouped product insights
✔ Visualization image with colored bounding boxes

🚀 Features

✔ YOLOv8 Object Detection

Detects retail products on shelves.

✔ ResNet50 Embedding Generation

Extracts high-dimensional features for similarity comparison.

✔ DBSCAN Clustering

Groups visually similar products.

✔ Visualization Engine

Creates output images with bounding boxes color-coded by cluster.

✔ REST API

Accepts an image → returns JSON + visualization.

🧠 Tech Stack Used
AI / ML

YOLOv8 (Ultralytics)

ResNet50 (TorchVision)

PyTorch

NumPy

Scikit-Learn (DBSCAN)

Backend API

Flask (Python)
