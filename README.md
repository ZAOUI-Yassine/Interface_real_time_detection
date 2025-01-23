# Real-Time Object Detection System

## Overview

This project focuses on developing a **real-time object detection system** capable of identifying a small set of common objects from a live video feed. The system detects objects such as a **cup**, **phone**, **bottle**, **fork**, and **spoon**, using computer vision and machine learning technologies. 

The objective is to provide an efficient and user-friendly tool for real-time object recognition, balancing **accuracy** and **speed** to ensure smooth performance even on devices with limited computational resources.

## Key Features

- **Real-Time Processing**: The system captures video streams from a computer's webcam and performs object detection on each frame.
- **Lightweight and Fast**: Optimized for low-latency detection using pre-trained models and quantization techniques.
- **Python-Based**: Leverages Python's extensive ecosystem for computer vision and machine learning tasks.

## Technologies Used

- **OpenCV**: Handles video capture, image processing, and annotation of detected objects.
- **MediaPipe**: Employs pre-trained models for object detection, specifically the `EfficientDet-Lite0` model for its balance of speed and accuracy.
- **NumPy**: Used for efficient manipulation of image data.
- **TensorFlow Lite Models**: Enables lightweight and fast inference for object detection.

## Objectives

- Create a streamlined system to detect and classify objects in real time.
- Maintain low latency (30-40 ms per frame) while ensuring high accuracy (confidence scores between 0.5 and 0.75).
- Use efficient quantized models to run smoothly on typical consumer-grade laptops.

## Future Improvements

- Integrate robustness testing with noisy or challenging data (e.g., poor lighting conditions).
- Customize the detection model for better accuracy on specific object classes through fine-tuning and data augmentation.
- Experiment with more advanced architectures like **YOLO** or **SSD** for improved performance in specific scenarios.

## Conclusion

This project demonstrates the development of a practical real-time object detection system, combining lightweight models and efficient libraries to achieve high performance on resource-constrained devices. It's a strong foundation for further exploration in computer vision applications.

---

Feel free to clone this repository and try the project on your own setup. Contributions and suggestions are always welcome!
