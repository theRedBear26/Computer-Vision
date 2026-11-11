import cv2
import requests
import numpy as np
from ultralytics import YOLO
import os

def download_image(url, filename):
    print(f"Downloading sample site image...")
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def detect_site_safety(image_path):
    # 1. Load the pre-trained YOLOv8 model
    # In a real deployment, you would train this model on a "Hard Hat & Vest" dataset
    # and load your custom weights: YOLO('best_hardhat_model.pt')
    model = YOLO('yolov8n.pt') 
    
    # 2. Run inference on the image
    results = model(image_path)
    
    # 3. Process results
    img = cv2.imread(image_path)
    violations = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get class ID (0 is 'person' in COCO dataset)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == 0 and conf > 0.5: # If it's a person with >50% confidence
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw a red box around the person
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"Worker: {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # LOGIC: Check if they have a hard hat (Mocked here)
                # In a real app, you'd check if a 'hardhat' box overlaps with this 'person' box.
                violations += 1
                
    print(f"✅ Analysis Complete. Detected {violations} workers on site.")
    print("⚠️  Safety Logic: None of them are wearing hard hats (Base model limitation).")
    
    # Save and show the image
    output_path = "site_safety_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"Saved output to {output_path}")
    
    # Display the image (Press 'q' to quit)
    cv2.imshow("Construction Site Safety", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Sample image of construction workers
    image_url = "https://images.unsplash.com/photo-1541888049184-67217995b981?q=80&w=1000"
    img_file = "test_site.jpg"
    
    if not os.path.exists(img_file):
        download_image(image_url, img_file)
        
    detect_site_safety(img_file)