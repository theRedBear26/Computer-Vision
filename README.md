# Computer Vision Journey

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Actively%20Learning-brightgreen)]()

A hands-on repository documenting my journey into **Computer Vision**,
from image processing fundamentals with OpenCV to classical computer
vision, deep learning, and real-world visual AI applications.

My goal is to understand how computers interpret images and videos,
build practical projects, and develop the engineering skills needed to
contribute to useful computer vision systems.

> Learning in public: this repository will grow as I study, experiment,
> build projects, and improve my implementations.

------------------------------------------------------------------------

## About this repository

This is both a learning notebook and a practical project portfolio. It
contains code, experiments, explanations, and projects covering:

-   Image loading, displaying, saving, and manipulation
-   Image transformations and geometric operations
-   Color spaces, thresholding, and image filtering
-   Edges, contours, and shape analysis
-   Feature detection and matching
-   Object detection, tracking, and video processing
-   Deep learning for image classification and object detection
-   Practical applications of computer vision

The code is organized by topic so that learners and reviewers can follow
the progression from fundamentals to more advanced methods.

## Technology stack

  -----------------------------------------------------------------------
  Technology                          How I use it
  ----------------------------------- -----------------------------------
  Python                              Main programming language

  OpenCV                              Image processing, video processing,
                                      and classical computer vision

  NumPy                               Image arrays and numerical
                                      operations

  Matplotlib                          Visualization and analysis

  PyTorch                             Deep learning experiments (as
                                      introduced)

  Ultralytics YOLO                    Object detection experiments (as
                                      introduced)

  Git and GitHub                      Version control and project
                                      documentation
  -----------------------------------------------------------------------

*The tools listed here describe the technologies used or planned for
this learning path; individual project folders document their actual
dependencies.*

## Repository structure

``` text
computer-vision/
├── resources/
│   ├── photos/
│   └── videos/
├── 01_opencv_basics/
├── 02_image_processing/
├── 03_classical_computer_vision/
├── 04_video_processing/
├── 05_feature_detection/
├── 06_deep_learning/
├── projects/
├── notebooks/
├── requirements.txt
└── README.md
```

The folder names are a suggested organization. I will update this tree
to match the actual contents of the repository as it develops.

## Learning roadmap

### 1. OpenCV fundamentals

-   Read, display, and save images
-   Understand image dimensions, channels, and pixel values
-   Work with image regions and resizing
-   Draw shapes, lines, and text
-   Capture and process video
-   Handle mouse and keyboard events

### 2. Image processing

-   Color spaces: BGR, RGB, HSV, and grayscale
-   Image thresholding: global, Otsu, and adaptive
-   Smoothing and filtering
-   Image gradients and edge detection
-   Morphological operations: erosion, dilation, opening, and closing
-   Histograms and contrast enhancement

### 3. Classical computer vision

-   Contours and connected components
-   Shape measurements and moments
-   Convex hulls and shape matching
-   Hough line and circle detection
-   Template matching
-   Corner and feature detection
-   Feature description and matching: SIFT, ORB, and AKAZE

### 4. Video and motion

-   Video capture and frame processing
-   Background subtraction
-   Object tracking
-   Optical flow
-   Motion detection

### 5. Deep learning for computer vision

-   Neural network fundamentals
-   Convolutional neural networks (CNNs)
-   Image classification
-   Object detection
-   Image segmentation
-   Transfer learning and fine-tuning
-   Model evaluation and error analysis

### 6. Deployment and real-world applications

-   Build reusable Python modules
-   Create APIs for inference
-   Process images and video efficiently
-   Test models on real-world data
-   Document limitations, performance, and responsible use

## Projects

Projects will be added here as they become functional and documented.

  -----------------------------------------------------------------------
  Project                 Description             Status
  ----------------------- ----------------------- -----------------------
  OpenCV fundamentals     Experiments with image  In progress
                          loading, display,       
                          drawing, and            
                          interaction             

  Image processing        Reusable examples of    Planned
  toolkit                 filtering,              
                          thresholding, and       
                          morphology              

  Shape and contour       Detect and measure      Planned
  analysis                shapes in images        

  Feature matching        Match visual features   Planned
                          between images          

  Object detection        Detect and localize     Planned
                          objects in images or    
                          video                   

  Real-world CV           A complete application  Planned
  application             built around a clearly  
                          defined problem         
  -----------------------------------------------------------------------

For each completed project, I aim to include its objective, sample
inputs and outputs, setup instructions, implementation details, and
known limitations.

## Getting started

### Requirements

-   Python 3.9 or newer
-   pip
-   Git

### 1. Clone the repository

``` bash
git clone https://github.com/YOUR_USERNAME/computer-vision.git
cd computer-vision
```

Replace `YOUR_USERNAME` with your GitHub username and adjust the
repository URL if your repository has a different name.

### 2. Create a virtual environment

Linux / macOS:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

``` powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install opencv-python numpy matplotlib
```

If a `requirements.txt` file is maintained for a particular project,
install its dependencies with:

``` bash
pip install -r requirements.txt
```

### 4. Run an example

``` python
import cv2

image = cv2.imread("resources/photos/example.jpg")

if image is None:
    raise FileNotFoundError("Could not load the image. Check the file path.")

cv2.imshow("Computer Vision", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Use an image that exists at the path you provide. GUI display may
require a desktop environment; for notebooks or headless systems, use an
appropriate display method instead.

## How I approach each project

For each substantial project, I aim to document:

1.  **Problem:** What visual task am I trying to solve?
2.  **Data:** Where do the images or videos come from, and what are
    their limitations?
3.  **Method:** Which image-processing or machine-learning techniques
    are used?
4.  **Implementation:** How is the solution structured?
5.  **Evaluation:** How well does it work, and how was that measured?
6.  **Limitations:** Where does it fail, and what should be improved?

I believe a good computer vision project should demonstrate not only
that a model runs, but also that the problem, data, evaluation, and
limitations are understood.

## Current focus

I am strengthening my foundations in Python, NumPy, and OpenCV, with a
focus on understanding image processing and classical computer vision
before moving deeper into neural networks and modern detection models.

I am especially interested in applying computer vision to practical
challenges in areas such as:

-   Agriculture and crop monitoring
-   Coffee quality inspection
-   Manufacturing and visual quality control
-   Transportation and traffic analysis
-   Healthcare research and medical imaging

These are areas of interest, not claims of deployed products or
validated solutions.

## Connect

I welcome feedback, learning resources, collaboration, and opportunities
to contribute to computer vision and AI projects.

-   **GitHub:** https://github.com/theRedBear26
-   **Email:** yilakbelayineh695@gmail.com

------------------------------------------------------------------------

⭐ If you find this learning repository useful, feel free to star it or
share suggestions. Thank you for following my computer vision journey!
