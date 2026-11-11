Civil Engineering x Computer Vision (CECV)
Applying Computer Vision and Deep Learning to Real-World Engineering Problems

I'm a Civil Engineer with a growing interest in Artificial Intelligence, particularly Computer Vision and Deep Learning. I started this project to explore how the skills I'm learning in AI can be applied to problems in civil engineering and construction.

Most of my projects focus on areas such as construction site safety, structural inspection, and construction progress monitoring. I wanted to work on problems that connect my engineering background with my interest in programming and machine learning.

This repository contains my practical experiments, projects, and implementations as I continue developing my skills in this field.

🏗️ Civil Engineering Applications
1. Construction Site Safety – PPE Detection

File: construction_Site_Safety_(PPE Detection).py

Technologies: Python, YOLOv8, OpenCV

Construction site safety is an important part of civil engineering. Workers are required to wear appropriate personal protective equipment (PPE), but manually checking every worker can be difficult, especially on large construction sites.

In this project, I used YOLOv8 to detect objects in a construction site image. The program identifies objects and displays the detection results using bounding boxes.

My goal with this project was to understand how object detection could be used as a starting point for automated PPE monitoring.

2. Infrastructure Inspection – Crack Segmentation

File: Infrastructure_Inspection_(Crack_Segmentation).py

Technologies: Python, PyTorch, U-Net

Cracks in concrete structures can be early signs of deterioration. Identifying and measuring them is important when assessing the condition of infrastructure.

For this project, I implemented a U-Net architecture using PyTorch to explore semantic segmentation for crack detection.

Unlike object detection, which identifies an object using a bounding box, segmentation aims to identify the individual pixels belonging to a crack.

I chose this approach because it can provide more detailed information about the location and shape of visible cracks.

3. Construction Progress Tracking – Transfer Learning

File: Construction_Progress_Tracking_(Transfer_Learning).py

Technologies: Python, TensorFlow, MobileNetV2

Keeping track of construction progress is essential for project planning and management. Site inspections and progress reporting can take considerable time.

In this project, I explored image classification using transfer learning with MobileNetV2.

The idea is to classify construction site images into different categories representing construction activities or stages, such as earthworks, framing, and heavy machinery.

This project helped me understand how pretrained deep learning models can be adapted to engineering-related image classification tasks.

4. Capstone Project – Concrete Crack Detection

Files:

Concrete_Crack_Detection.py
Capstone_Deployed_System_(Streamlit).py

Technologies: Python, TensorFlow, CNN, Streamlit

This is one of the projects where I brought together model development and application deployment.

I developed a CNN-based approach for concrete crack classification and created a Streamlit application that allows users to upload an image and receive a model prediction.

The purpose was to go beyond training a model in a notebook and understand how a machine learning model can be made accessible through a simple web application.

🧠 Deep Learning Implementations from Scratch

Alongside my engineering-related projects, I have been studying how neural networks and modern deep learning architectures work internally.

I included these implementations to strengthen my understanding of the mathematics and concepts behind deep learning.

Neural Network from Scratch

Files:

Neural_Network_from_Scratch.py
Neural_Network_from_Scratch_(NumPy).py

Implemented the basic components of a neural network using NumPy, including:

Forward propagation
Backpropagation
Chain rule and gradient calculations
Gradient descent

I experimented with training the network on the XOR problem and the 8×8 digits dataset.

Transformer from Scratch

File: Transformer_from_Scratch_in_PyTorch.py

Technology: PyTorch

Implemented a Transformer Encoder architecture to understand the main components behind the Transformer model.

The implementation covers:

Positional encoding
Multi-head self-attention
Transformer encoder components
Sequence reversal as a training task
Mini GPT – Character-Level Language Model

File: Mini_GPT_(Character-Level).py

Technology: PyTorch

A small character-level language model based on the GPT architecture.

This project helped me explore causal self-attention, masked attention, token sequences, and autoregressive text generation.

💻 Computer Vision and Machine Learning Fundamentals

I also included smaller projects that helped me build a foundation in computer vision and machine learning.

Project	Description
Linear_Regression_from_Scratch.py	Implements linear regression using gradient descent and NumPy, including loss visualization.
detecting_mouseclick.py	Uses OpenCV mouse events to select points and perform perspective transformation on an image.
how_tocrate_black_images_colored_images_shapes_and_text.py	Explores basic OpenCV operations, including image creation, drawing shapes, and adding text.
🛠️ Technologies and Tools
Programming: Python, NumPy
Deep Learning: PyTorch, TensorFlow
Computer Vision: OpenCV
Object Detection: Ultralytics YOLOv8
Web Application: Streamlit
Machine Learning: CNNs, U-Net, Transfer Learning, Transformers
Development: Git, GitHub
🚀 Running the Projects Locally
1. Clone the repository
bash
git clone https://github.com/theRedBear26/Computer-Vision.git
cd Computer-Vision

2. Create a virtual environment
bash
python -m venv venv


Activate it:

Windows

bash
venv\Scripts\activate


Linux / macOS

bash
source venv/bin/activate

3. Install dependencies

Install the libraries required by the particular project:

bash
pip install numpy opencv-python torch torchvision tensorflow streamlit ultralytics matplotlib

4. Run a project

For example:

bash
python "construction_Site_Safety_(PPE Detection).py"


For the Streamlit application:

bash
streamlit run "Capstone_Deployed_System_(Streamlit).py"


Note: Some projects may require additional datasets, image files, or model weights. Check the individual scripts before running them.

🎯 What I'm Working Towards

My long-term goal is to develop my skills at the intersection of civil engineering, computer vision, and artificial intelligence.

I'm particularly interested in exploring how AI can support:

Structural condition assessment and defect detection.
Construction site monitoring and safety.
Infrastructure inspection.
Construction progress monitoring.
Data-driven engineering decisions.

I'm still learning, experimenting, and improving my implementations. This repository represents part of that journey.

I'm open to internship opportunities where I can contribute my civil engineering knowledge, develop my programming skills, learn from experienced engineers and developers, and work on practical problems.

Thank you for taking the time to look through my work!