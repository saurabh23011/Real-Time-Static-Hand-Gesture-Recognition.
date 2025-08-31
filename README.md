# Real-Time Hand Gesture Recognition Application

## Information About me 
Project Created by Saurabh Singh and email: saurabhsingh802213@gmail.com 

## Project Demo is :


https://github.com/user-attachments/assets/7c21f17f-4b13-427e-b0be-3fe4e3044fe3













## Technology Justification
For this hand detection and gesture recognition application, I selected MediaPipe as the primary framework for hand landmark detection, combined with OpenCV (cv2) for video capture and processing, and NumPy for numerical computations. 

MediaPipe was chosen because it provides a high-fidelity, real-time solution for hand and finger tracking, inferring 21 3D landmarks from a single frame using machine learning without requiring custom training for basic gestures. and  it excels in efficiency, running on a CPU with high FPS, making it ideal for webcam-based applications like this one. Pre-trained models handle detection and tracking robustly, supporting applications such as gesture recognition and sign language interpretation. Alternatives like OpenPose were considered, but MediaPipe is lighter-weight and more optimized for edge devices, offering better real-time performance without heavy dependencies. YOLOv7 Pose is another option for pose estimation, but it may struggle with fine-grained hand details compared to MediaPipe's specialized hand models. OpenCV complements MediaPipe by handling webcam input, image flipping, and drawing overlays, while NumPy supports vector calculations for distances and angles. This stack ensures simplicity, accuracy (up to 99.95% in similar setups), and cross-platform compatibility.

## Gesture Logic Explanation
The application recognizes five gestures (though the assignment specifies four required; this includes an extra for robustness): Fist ✊, Peace Sign ✌️, Thumbs Up 👍, Open Palm ✋, and Pointing 👉. Detection relies on MediaPipe's hand landmarks to determine finger states (extended or not) based on y-coordinate comparisons for alignment (stricter for non-thumb fingers) and x-coordinates for the thumb.

- **Fist ✊**: Triggers if no fingers are extended (extended_count == 0).
- **Peace Sign ✌️**: Only index and middle fingers extended, with a minimum distance (>0.02) between their tips to ensure separation.
- **Thumbs Up 👍**: Only thumb extended and pointing upward (thumb tip y < wrist y).
- **Open Palm ✋**: All five fingers extended (extended_count == 5).
- **Pointing 👉**: Only index finger extended.

Smoothing uses a history buffer to confirm gestures over multiple frames, reducing flicker.

## Setup and Execution Instructions
1. Ensure Python 3.8+ is installed.
2. Clone the repository or save the code as `gesture_recognizer.py`.
3. Create and activate a virtual environment: `python -m venv env` then `source env/bin/activate` (Linux/Mac) or `env\Scripts\activate` (Windows).
4. Install dependencies: `pip install -r requirements.txt`.
5. Run the application: `python gesture_recognizer.py`.
6. Use webcam; press 'q' to quit, 's' for screenshot, 'r' to reset history.
7. Allow camera access; test in good lighting for best results.

## Demonstration
View a short demo video showing similar gesture recognition in action: [Hand Tracking with MediaPipe and OpenCV](https://www.youtube.com/watch?v=RRBXVu5UE-U). This illustrates Fist, Open Palm, Thumbs Up, and Pointing (Peace Sign is analogous). Run the code locally for your own webcam demo.
