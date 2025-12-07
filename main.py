import cv2
import time
import numpy as np
import os

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception as e:
    print(f"Warning: DeepFace could not be imported. Emotion detection will be disabled. Error: {e}")
    DEEPFACE_AVAILABLE = False

class FaceAnalyzer:
    def __init__(self):
        # Initialize OpenCV Face Detection (Haar Cascade)
        # This is lighter and works on all python versions
        # Load Haar Cascade from local file for stability
        cascade_filename = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_filename):
            print(f"Error: {cascade_filename} not found in current directory!")
            # Try to download if missing (simple fallback)
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            print(f"Downloading {cascade_filename} from {url}...")
            urllib.request.urlretrieve(url, cascade_filename)
            
        print(f"Loading Face Cascade from local file: {cascade_filename}")
        self.face_cascade = cv2.CascadeClassifier(cascade_filename)
        
        if self.face_cascade.empty():
             print("CRITICAL ERROR: Failed to load local haarcascade. Face detection will not work.")
        else:
             print("Success: Loaded local Face Cascade XML correctly.")

        
        # Performance optimization hints
        print("Loading Emotion Model...")
        if DEEPFACE_AVAILABLE:
            try:
                # We use a dummy image. 
                # Note: DeepFace might try to download weights on first run.
                DeepFace.analyze(img_path = np.zeros((48, 48, 3), dtype=np.uint8), actions = ['emotion'], enforce_detection=False)
                print("Model Loaded!")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("DeepFace not available. Skipping model load.")

    def detect_faces(self, image):
        """
        Detects faces in the image using OpenCV Haar Cascades.
        """
        try:
            # Haar cascades work on grayscale images
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if self.face_cascade.empty():
                print("Debug: Face cascade object is empty during loop.")
                return []

            # Detect faces
            # Try lower minNeighbors to be more aggressive
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            
            if len(faces) > 0:
                print(f"Debug: Detected {len(faces)} faces.")
                
            # Convert to list of (x, y, w, h)
            rects = []
            for (x, y, w, h) in faces:
                rects.append((x, y, w, h))
                
            return rects
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []

    def analyze_emotion(self, face_img):
        """
        Analyzes emotion for a cropped face image.
        """
        if not DEEPFACE_AVAILABLE:
            return "N/A", {"N/A": 0.0}

        try:
            # detector_backend='skip' because we already have the face crop
            objs = DeepFace.analyze(img_path = face_img, 
                                  actions = ['emotion'], 
                                  enforce_detection = False,
                                  detector_backend = 'skip')
            
            if objs:
                # DeepFace returns a list of result objects
                return objs[0]['dominant_emotion'], objs[0]['emotion']
        except Exception as e:
            # print(f"Error in analysis: {e}")
            pass
        return None, None

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    analyzer = FaceAnalyzer()
    
    # Video Writer for recording
    # We use XVID codec for AVI format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Output file: output_preview.avi, 20.0 FPS, Resolution (640, 480) - Adjust if needed
    out = cv2.VideoWriter('output_preview.avi', fourcc, 20.0, (640, 480))
    
    print("Recording started. Saving to 'output_preview.avi'. Press 'Esc' to stop.")
    
    # Variables for FPS calculation
    prev_frame_time = time.time()
    new_frame_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Resize to match VideoWriter resolution
        frame = cv2.resize(frame, (640, 480))

        # Detect Faces
        faces = analyzer.detect_faces(frame)

        for (x, y, w, h) in faces:
            # Always draw bounding box for detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Analyze Emotion (only for faces larger than 40px)
            if w > 40 and h > 40:
                face_roi = frame[y:y+h, x:x+w]
                emotion, emotion_scores = analyzer.analyze_emotion(face_roi)

                if emotion and emotion_scores:
                    # Display emotion text with background for better visibility
                    text = f"{emotion} ({int(emotion_scores[emotion])}%)"
                    
                    # Draw background rectangle for text
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (x, y - 30), (x + text_size[0] + 10, y), (0, 255, 0), -1)
                    cv2.putText(frame, text, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                else:
                    # Show "Analyzing..." if emotion not detected yet
                    cv2.putText(frame, "Analyzing...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if new_frame_time - prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
        
        # Write the frame to the video file
        out.write(frame)

        cv2.imshow('Face Emotion Recognition Data', frame)

        if cv2.waitKey(1) & 0xFF == 27: # Press 'Esc' to exit
            break

    cap.release()
    out.release() # Save video
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
