import av
import cv2
import time
from services.yolo_service import YoloService

class VideoProcessor:
    """
    Handles WebRTC frame-by-frame processing.
    Now accepts dynamic parameters (confidence, box styles, privacy mode) 
    injected live from the Streamlit main thread.
    """
    def __init__(self):
        self.yolo_service = YoloService("Model/ppe.pt")
        self.last_hardhat_time = time.time()
        self.alert_cooldown_seconds = 5
        self.violation_active = False
        
        # Dynamic interactive attributes (Default values)
        self.conf_threshold = 0.25
        self.selected_classes = ["Hardhat", "Safety Vest", "Person", "NO-Hardhat", "NO-Safety Vest", "Machinery", "Vehicle"]
        self.box_style = "Standard"
        self.privacy_mode = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Standardize media sizing for the UI constraint
        img = cv2.resize(img, (854, 480))
        
        # Execute parameterized inference
        annotated_img, stats, hardhat_detected, person_detected, _, _ = self.yolo_service.predict_and_annotate(
            frame=img,
            conf_threshold=self.conf_threshold,
            selected_classes=self.selected_classes,
            box_style=self.box_style,
            privacy_mode=self.privacy_mode
        )
        
        current_time = time.time()

        if hardhat_detected:
            self.last_hardhat_time = current_time
            self.violation_active = False
            
        time_since_hardhat = current_time - self.last_hardhat_time
        
        if person_detected and not hardhat_detected and time_since_hardhat >= self.alert_cooldown_seconds:
            self.violation_active = True
                
        self._overlay_dashboard(annotated_img, stats, self.violation_active)

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        
    def _overlay_dashboard(self, frame, stats, violation_active):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (280, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"Hardhats: {stats.get('Hardhat', 0)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vests: {stats.get('Safety Vest', 0)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"People: {stats.get('Person', 0)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if violation_active:
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 255), -1)
            cv2.putText(frame, "CRITICAL ALERT: MISSING HARDHAT!", (50, h-30), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
