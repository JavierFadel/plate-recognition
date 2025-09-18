# Jakarta Street CCTV License Plate Recognition System

A comprehensive computer vision project specifically designed for Indonesian traffic scenarios that automatically detects vehicles and extracts license plate text from CCTV street images using state-of-the-art deep learning models.

## Project Overview

This project implements an end-to-end license plate recognition system optimized for Indonesian traffic conditions, specifically targeting:
- **Jakarta Street CCTV Images**: Designed for real-world CCTV footage from Jakarta streets
- **Weather Variations**: Robust performance across different weather conditions (sunny, rainy, cloudy)
- **Day/Night Scenarios**: Optimized for both daylight and nighttime CCTV captures
- **Indonesian License Plates**: Tailored for Indonesian vehicle registration plate formats

### Key Capabilities
- **Vehicle Detection**: Using YOLOv8 for accurate vehicle identification in CCTV angles
- **License Plate Localization**: Computer vision techniques optimized for street-level camera perspectives
- **Text Recognition**: OCR specifically tuned for Indonesian license plate characters
- **Image Enhancement**: Advanced preprocessing to handle varying lighting and weather conditions

The system processes batch CCTV images from Jakarta streets, automatically identifying license plate numbers across diverse environmental conditions, making it suitable for traffic monitoring, law enforcement, and urban management applications.

## Technology Stack

### Core Technologies
- **Python 3.x** - Primary programming language
- **Jupyter Notebook** - Development environment
- **OpenCV (cv2)** - Computer vision and image processing
- **YOLOv8 (Ultralytics)** - Object detection for vehicle identification
- **EasyOCR** - Optical Character Recognition for text extraction

### Libraries & Dependencies
```bash
# Core Dependencies
pip install ultralytics     # YOLOv8 implementation  
pip install easyocr         # OCR engine for text recognition
pip install opencv-python   # Computer vision operations
pip install numpy          # Numerical computing
pip install matplotlib     # Data visualization and plotting

# Optional: For additional OCR capabilities
pip install keras-ocr      # Alternative OCR framework (optional)
```

### Machine Learning Models
- **YOLOv8n**: Nano version of YOLOv8 for efficient vehicle detection in CCTV footage
- **EasyOCR**: Pre-trained OCR model with English language support (suitable for Indonesian plates)

## System Architecture

### Processing Pipeline

1. **Image Input**: Load images from specified directory
2. **Vehicle Detection**: 
   - Use YOLOv8 to detect vehicles in the image
   - Apply confidence threshold (0.5) to filter detections
3. **Vehicle Cropping**: Extract regions of interest containing vehicles
4. **License Plate Detection**:
   - Convert to grayscale
   - Apply Gaussian blur for noise reduction
   - Use Canny edge detection
   - Find contours with specific aspect ratio criteria (2:1 to 5:1)
5. **Image Enhancement**:
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Detail enhancement for better OCR performance
6. **Text Extraction**: Apply EasyOCR to extract text from plate regions
7. **Visualization**: Display results with bounding boxes and detected text

### Key Functions

#### `enhance_image(image)`
Enhances image quality for better OCR performance:
- Converts to grayscale
- Applies CLAHE for contrast enhancement
- Uses detail enhancement filter

#### `detect_plate_from_vehicle(cropped_img)`
Detects potential license plate regions within vehicle crops:
- Edge detection using Canny algorithm
- Contour analysis with aspect ratio filtering
- Returns candidate plate regions

## Features

- **Multi-vehicle Detection**: Processes multiple vehicles in a single image
- **Robust Plate Detection**: Uses computer vision techniques to locate plates accurately
- **Image Enhancement**: Preprocessing pipeline to improve OCR accuracy
- **Batch Processing**: Processes entire directories of images automatically
- **Visual Output**: Displays detection results with bounding boxes and extracted text
- **Grid Visualization**: Shows detected license plates in an organized grid layout

## Usage

### Environment Setup

This project can be run in multiple environments:

#### Option 1: Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics easyocr opencv-python numpy matplotlib
```

#### Option 2: Google Colab
```python
# Install packages in Colab
!pip install ultralytics easyocr opencv-python

# Mount Google Drive for dataset access
from google.colab import drive
drive.mount('/content/drive')
```

#### Option 3: Conda Environment
```bash
# Create conda environment
conda create -n plate-recognition python=3.9
conda activate plate-recognition

# Install dependencies
pip install ultralytics easyocr opencv-python numpy matplotlib
```

### Dataset Preparation

1. **Organize your Jakarta CCTV images**:
```
your-dataset/
├── image1.jpg
├── image2.jpg
└── ...
```

2. **Update dataset path** in the notebook:
```python
# For local development
data_folder = '/path/to/your/dataset'

# For Google Colab
data_folder = '/content/drive/MyDrive/your-dataset-folder'
```

### Running the Analysis

1. **Open the Jupyter notebook**:
```bash
jupyter notebook main.ipynb
```

2. **Execute cells sequentially**:
   - Install dependencies
   - Configure dataset path
   - Run the detection pipeline

### Expected Input
- **Image Format**: JPG, PNG, or other OpenCV-supported formats
- **Content**: Jakarta street CCTV images containing vehicles with visible license plates
- **Conditions**: Various weather (sunny/rainy) and lighting (day/night) conditions
- **Organization**: Images stored in a single directory

### Output
- Vehicle detection bounding boxes overlaid on original images
- Grid display of extracted license plate regions
- Text recognition results for each detected plate

## Configuration

### Detection Parameters
```python
# YOLO confidence threshold
conf=0.5

# License plate aspect ratio filters
aspect_ratio > 2 and aspect_ratio < 5

# Minimum plate dimensions
w > 50 and h > 15

# CLAHE parameters
clipLimit=2.0
tileGridSize=(8, 8)

# Detail enhancement parameters
sigma_s=10
sigma_r=0.15
```

## 📊 Performance Considerations

### Optimization Features
- **YOLOv8n**: Uses the nano version for faster inference
- **Confidence Filtering**: Reduces false positives
- **Aspect Ratio Filtering**: Focuses on plate-like shapes
- **Batch Processing**: Efficient handling of multiple images

### Limitations
- **Language Support**: Currently optimized for English text
- **Lighting Conditions**: Performance may vary in extreme lighting
- **Plate Orientation**: Works best with front-facing plates
- **Image Quality**: Requires reasonably clear images for accurate OCR

## Troubleshooting

### Common Issues
1. **No vehicles detected**: Check image quality and YOLO confidence threshold
2. **OCR accuracy**: Ensure adequate image resolution and contrast
3. **Missing dependencies**: Verify all packages are installed correctly
4. **Google Drive access**: Ensure proper mounting and file permissions

## Future Enhancements

- **Multi-language OCR**: Support for additional languages
- **Real-time Processing**: Video stream processing capability
- **Database Integration**: Store and manage detected plate data
- **API Development**: REST API for integration with other systems
- **Mobile Deployment**: Lightweight version for mobile devices

## Dataset Information

### Jakarta Street CCTV Dataset
This project is specifically designed for the Jakarta street CCTV dataset with the following characteristics:

**Dataset Features:**
- **Location**: Street intersections and roads across Jakarta, Indonesia
- **Camera Angle**: CCTV street-level perspective (typically elevated 4-6 meters)
- **Weather Variations**: 
  - Sunny conditions with high contrast
  - Rainy weather with reduced visibility
  - Cloudy/overcast lighting
- **Time Variations**:
  - Daylight images (bright, high contrast)
  - Nighttime images (artificial lighting, reflections)
- **Vehicle Types**: Motorcycles, cars, trucks, and buses
- **Traffic Density**: Various traffic conditions from light to heavy congestion

**Image Naming Convention:**
```
BXS_AUS_270213022102_20240619101203663_X589Y488W36H10_Motorcycle_Audi_unknown_066_01_09724.jpg
```

**Metadata Structure:**
- **Location Code**: BXS_AUS (Jakarta area identifier)
- **Timestamp**: Date and time of capture
- **Coordinates**: X, Y position and width, height
- **Vehicle Type**: Motorcycle, Car, Truck, etc.
- **Additional Info**: Brand, sequence numbers

**Challenges Addressed:**
- **Weather Adaptability**: Enhanced preprocessing for rainy/foggy conditions
- **Lighting Variations**: CLAHE and detail enhancement for night images
- **Indonesian Plate Format**: Optimized for local license plate characteristics
- **CCTV Perspective**: Adjusted detection parameters for elevated camera angles

## Version Information

- **Development Environment**: Google Colab / Jupyter Notebook
- **Python Version**: 3.x
- **YOLOv8**: Latest Ultralytics implementation
- **OpenCV**: Latest stable version
- **EasyOCR**: Latest stable version