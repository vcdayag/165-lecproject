import urllib.request

def downloadModel():
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    urllib.request.urlretrieve(url, "./yolov8n.pt")

if __name__ == "__main__":
    downloadModel()