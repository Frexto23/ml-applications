import detection

while True:
    img_url = input()

    detector = detection.ImgObjectDetector()
    detector.detect_and_show(img_url)
