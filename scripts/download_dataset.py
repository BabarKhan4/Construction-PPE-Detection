import os
from roboflow import Roboflow

def download_dataset():
    print("Initializing Roboflow...")
    rf = Roboflow(api_key="ovEadVpHdxTS0IOKDsxy")
    
    print("Fetching workspace and project...")
    workspace = rf.workspace("ppe-detection-khsge")
    project = workspace.project("ppe-a6gr5")
    
    print("Downloading dataset (YOLOv8 format)...")
    # Setting the download location to a 'datasets' folder in the project root
    os.makedirs("datasets", exist_ok=True)
    os.chdir("datasets")
    dataset = project.version(1).download("yolov8")
    
    print(f"Dataset successfully downloaded to: {os.path.abspath(os.getcwd())}")

if __name__ == "__main__":
    download_dataset()
