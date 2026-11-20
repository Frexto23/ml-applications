from io import BytesIO
import torch
import requests
import matplotlib.pyplot as plt
from PIL import Image

class ImgObjectDetector:
    def __init__(self):
        self._model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        device = 'cuda'
        self._model.to(device)

    def process_image(self, image):
        results = self._model(image)
        return results.render()[0]

    def detect_and_show(self, image_url):
        response = requests.get(image_url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert('RGB')

        rendered_image = self.process_image(image)

        plt.figure(figsize=(12, 8))
        plt.imshow(rendered_image)
        plt.axis("off")
        plt.show()