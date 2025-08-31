
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import sys

class HandGestureRecognizer:
    """
    A class to handle real-time hand gesture recognition using MediaPipe
    """
    
    def __init__(self):
        """Initialize MediaPipe hands module and drawing utilities"""
        # Initialize MediaPipe hands with optimized settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,  # Increased for better detection
            min_tracking_confidence=0.7    # Increased for stability
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize gesture history for smoothing
        self.gesture_history = deque(maxlen=7)  # Increased buffer for more stability
        
        # FPS calculation variables
        self.fps_start_time = time.time()
        self.fps_counter = 0
        self.fps = 0
        
    def is_finger_extended(self, landmarks, finger_tip_id, finger_mcp_id, finger_pip_id=None):
        """
        Determine if a finger is extended based on landmark positions
        
        Args:
            landmarks: Hand landmarks
            finger_tip_id: Landmark ID for fingertip
            finger_mcp_id: Landmark ID for MCP joint
            finger_pip_id: Landmark ID for PIP joint (optional)
            
        Returns:
            Boolean indicating if finger is extended
        """
        tip = landmarks.landmark[finger_tip_id]
        mcp = landmarks.landmark[finger_mcp_id]
        
        if finger_pip_id:
            pip = landmarks.landmark[finger_pip_id]
            # For fingers other than thumb, check if tip is above PIP and PIP above MCP for straight extension
            return tip.y < pip.y and pip.y < mcp.y  # y decreases upward; stricter check for alignment
        else:
            # For thumb, use different logic based on x-coordinate
            return tip.x > mcp.x if landmarks.landmark[4].x > landmarks.landmark[17].x else tip.x < mcp.x
    
    def count_extended_fingers(self, landmarks):
        """
        Count how many fingers are extended
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Dictionary with finger states and count
        """
        # Corrected MediaPipe hand landmark indices for accuracy
        finger_states = {
            'thumb': self.is_finger_extended(landmarks, 4, 2, None),   # Thumb: tip 4, MCP 2
            'index': self.is_finger_extended(landmarks, 8, 5, 6),      # Index: tip 8, MCP 5, PIP 6
            'middle': self.is_finger_extended(landmarks, 12, 9, 10),   # Middle: tip 12, MCP 9, PIP 10
            'ring': self.is_finger_extended(landmarks, 16, 13, 14),    # Ring: tip 16, MCP 13, PIP 14
            'pinky': self.is_finger_extended(landmarks, 20, 17, 18)    # Pinky: tip 20, MCP 17, PIP 18
        }
        
        extended_count = sum(finger_states.values())
        
        return finger_states, extended_count
    
    def calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points
        
        Args:
            p1, p2, p3: Points as [x, y] coordinates
            
        Returns:
            Angle in degrees
        """
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def recognize_gesture(self, landmarks):
        """
        Recognize hand gesture based on landmark positions
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            String representing the detected gesture
        """
        if not landmarks:
            return "No Hand"
        
        # Get finger states
        finger_states, extended_count = self.count_extended_fingers(landmarks)
        
        # Additional measurements for specific gestures
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_tip = landmarks.landmark[12]
        ring_tip = landmarks.landmark[16]
        pinky_tip = landmarks.landmark[20]
        wrist = landmarks.landmark[0]
        
        # Calculate distance between index and middle fingertips (for peace sign)
        index_middle_dist = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
        
        # GESTURE RECOGNITION LOGIC
        
        # 1. PEACE SIGN: Only index and middle fingers extended with some separation
        if (finger_states['index'] and finger_states['middle'] and 
            not finger_states['ring'] and not finger_states['pinky'] and
            index_middle_dist > 0.02):  # Fingers should be separated
            return "Peace Sign "
        
        # 2. THUMBS UP: Only thumb extended, pointing upward
        if (finger_states['thumb'] and not finger_states['index'] and 
            not finger_states['middle'] and not finger_states['ring'] and 
            not finger_states['pinky'] and thumb_tip.y < wrist.y):
            return "Thumbs Up "
        
        # 3. FIST: No fingers extended
        if extended_count == 0:
            return "Fist "
        
        # 4. OPEN PALM: All fingers extended
        if extended_count == 5:
            return "Open Palm "
        
        # 5. INDEX POINTING: Only index finger extended
        if (finger_states['index'] and not finger_states['middle'] and 
            not finger_states['ring'] and not finger_states['pinky']):
            return "Pointing "
        
        # Default case
        return "Unknown"
    
    def smooth_gesture(self, gesture):
        """
        Smooth gesture recognition using a history buffer with confidence scoring
        
        Args:
            gesture: Current detected gesture
            
        Returns:
            Most stable gesture from recent history
        """
        self.gesture_history.append(gesture)
        
        if len(self.gesture_history) >= 4:  # Need at least 4 samples
            # Count occurrences of each gesture
            gesture_counts = {}
            for g in self.gesture_history:
                gesture_counts[g] = gesture_counts.get(g, 0) + 1
            
            # Find the most frequent gesture
            most_frequent = max(gesture_counts.items(), key=lambda x: x[1])
            
            # Only return if it appears in at least 60% of recent frames
            if most_frequent[1] >= len(self.gesture_history) * 0.6:
                return most_frequent[0]
        
        return gesture
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
        return self.fps
    
    def draw_info(self, image, gesture, hand_landmarks):
        """
        Draw gesture text and additional information on the image
        
        Args:
            image: Input image
            gesture: Detected gesture name
            hand_landmarks: MediaPipe hand landmarks
        """
        height, width = image.shape[:2]
        
        # Create a more prominent background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Choose color based on gesture recognition confidence
        text_color = (0, 255, 0) if gesture not in ["Unknown", "No Hand"] else (0, 255, 255)
        
        # Draw gesture text with larger font
        cv2.putText(image, f"Gesture: {gesture}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
        
        # Draw FPS
        fps = self.calculate_fps()
        cv2.putText(image, f"FPS: {fps}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw confidence indicator
        if gesture not in ["Unknown", "No Hand"]:
            cv2.circle(image, (350, 45), 12, (0, 255, 0), -1)
            cv2.putText(image, "OK", (340, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Draw hand landmarks and connections with better visibility
        if hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)
            )
    
    def run(self):
        """Main loop for running the hand gesture recognition system"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            print("Please check if your camera is connected and not being used by another application")
            return
        
        # Set camera properties for optimal performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
        
        print("Hand Gesture Recognition System Started")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit the application")
        print("  's' - Save a screenshot")
        print("  'r' - Reset gesture history")
        print("\nRecognizable Gestures:")
        print("  1. Open Palm ✋ - All fingers extended")
        print("  2. Fist ✊ - All fingers closed")
        print("  3. Peace Sign ✌️ - Index and middle fingers extended")
        print("  4. Thumbs Up 👍 - Only thumb extended upward")
        print("  5. Pointing 👉 - Only index finger extended")
        print("=" * 50)
        
        # Give camera time to initialize
        time.sleep(2)
        
        while True:
            try:
                # Read frame from webcam
                success, image = cap.read()
                if not success:
                    print("Warning: Failed to read from webcam")
                    continue
                
                # Flip image horizontally for selfie-view display
                image = cv2.flip(image, 1)
                
                # Improve image quality
                image = cv2.GaussianBlur(image, (5, 5), 0)
                
                # Convert BGR to RGB for MediaPipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image with MediaPipe
                results = self.hands.process(image_rgb)
                
                # Initialize gesture
                gesture = "No Hand"
                
                # Check if hand is detected
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Recognize gesture
                    raw_gesture = self.recognize_gesture(hand_landmarks)
                    
                    # Smooth gesture recognition
                    gesture = self.smooth_gesture(raw_gesture)
                    
                    # Draw information on image
                    self.draw_info(image, gesture, hand_landmarks)
                else:
                    # No hand detected - clear history to avoid false positives
                    self.gesture_history.clear()
                    self.draw_info(image, gesture, None)
                
                # Display the image
                cv2.imshow('Hand Gesture Recognition', image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"gesture_screenshot_{timestamp}.png"
                    cv2.imwrite(filename, image)
                    print(f"Screenshot saved as {filename}")
                elif key == ord('r'):
                    # Reset gesture history
                    self.gesture_history.clear()
                    print("Gesture history reset")
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        
        # Cleanup
        print("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

def main():
    """Main function to run the application"""
    try:
        # Create and run the gesture recognizer
        recognizer = HandGestureRecognizer()
        recognizer.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()