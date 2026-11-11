import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import matplotlib.pyplot as plt
from PIL import Image
import io

def download_and_load_image(url):
    print("Downloading construction site image...")
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert('RGB')
    return img

def analyze_site_progress(url):
    # 1. Load pre-trained MobileNetV2 model
    # In a real scenario, you'd replace the top layer and fine-tune on your 
    # own dataset of "Foundation", "Framing", "Finishing" phases.
    model = MobileNetV2(weights='imagenet')
    
    # 2. Load and preprocess the image
    img = download_and_load_image(url)
    img_resized = img.resize((224, 224))
    
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # 3. Predict
    print("Analyzing construction phase...")
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=5)[0]
    
    # 4. Display Results
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Site Image Analyzed for Progress")
    
    print("\n--- 🏗️ Construction Progress Analysis ---")
    for i, (imagenet_id, label, score) in enumerate(decoded):
        print(f"{i + 1}. {label.capitalize()}: {score*100:.2f}%")
        
    print("\n💡 Engineering Interpretation:")
    top_label = decoded[0][1].lower()
    if any(word in top_label for word in ['crane', 'scaffold', 'building', 'bridge']):
        print("-> Structure detected. Phase: Heavy Construction / Framing.")
    elif any(word in top_label for word in ['sand', 'dirt', 'trench', 'excavator']):
        print("-> Earthworks detected. Phase: Site Prep / Foundation.")
    else:
        print("-> General site context. Further inspection required.")
        
    plt.show()

if __name__ == "__main__":
    # Sample image of a building under construction
    sample_url = "https://images.unsplash.com/photo-1503387762-592deb58ef4e?q=80&w=1000"
    analyze_site_progress(sample_url)