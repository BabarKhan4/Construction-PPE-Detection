import cv2
import numpy as np
from ultralytics import YOLO

class YoloService:
    def __init__(self, model_path="Model/ppe.pt"):
        """Initialize the YOLO model for PPE detection."""
        self.model = YOLO(model_path)
        self.colors = [
            (255, 100, 100),  # Hardhat (Blue)
            (100, 255, 100),  # Mask (Green)
            (50, 50, 255),    # NO-Hardhat (Red)
            (255, 255, 100),  # NO-Mask (Cyan)
            (255, 100, 255),  # NO-Safety Vest (Magenta)
            (100, 255, 255),  # Person (Yellow)
            (200, 100, 255),  # Safety Cone (Purple)
            (200, 200, 100),  # Safety Vest (Olive)
            (100, 200, 200),  # Machinery (Teal)
            (150, 150, 150)   # Vehicle (Gray)
        ]

    def _draw_minimalist(self, frame, x1, y1, x2, y2, color, thickness=2, length=20):
        """Draws viewfinder-style corner brackets instead of a full box."""
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

    def _draw_cyberpunk(self, frame, x1, y1, x2, y2, color):
        """Draws a glowing neon box."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 6)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    def predict_and_annotate(self, frame, conf_threshold=0.25, selected_classes=None, box_style="Standard", privacy_mode=False):
        """
        Runs YOLO inference with highly customizable visualization constraints.
        Returns: annotated_frame, stats_dict, hardhat_detected_bool, person_detected_bool, event_logs_list, violations_crops_list
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        stats = {"Hardhat": 0, "Safety Vest": 0, "Person": 0}
        hardhat_detected = False
        person_detected = False
        
        event_logs = []
        violation_crops = []

        if results[0].boxes is not None:
            # We sort boxes to process "Person" first to determine privacy blurring
            boxes = sorted(results[0].boxes, key=lambda b: int(b.cls[0].item()))
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                name = self.model.names[cls_id]
                
                # Filter by active classes if specified
                if selected_classes is not None and name not in selected_classes:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                color = self.colors[cls_id % len(self.colors)]
                label = f"{name} {int(confidence*100)}%"
                
                # Logic Triggers
                if name in stats:
                    stats[name] += 1
                if name == "Hardhat":
                    hardhat_detected = True
                elif name == "Person":
                    person_detected = True
                    
                # Privacy Blurring Execution (Blur people if missing hardhat)
                # In a robust system we check IoU of Person vs Hardhat. Here we utilize the global scene state for the demo.
                is_violation = (name == "Person" and not hardhat_detected)
                
                if privacy_mode and is_violation:
                    roi = frame[y1:y2, x1:x2]
                    # Apply intense Gaussian Blur to face/body
                    if roi.size > 0:
                        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                        frame[y1:y2, x1:x2] = blurred_roi
                        
                if is_violation and name == "Person":
                    # Crop violation for the dashboard feed
                    try:
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size > 0:
                            violation_crops.append(crop)
                    except:
                        pass
                
                # Draw Visual Bounding Boxes
                if box_style == "Cyberpunk":
                    self._draw_cyberpunk(frame, x1, y1, x2, y2, color)
                elif box_style == "Minimalist":
                    self._draw_minimalist(frame, x1, y1, x2, y2, color)
                else: # Standard
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                
                if box_style == "Cyberpunk":
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0,0,0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, color, thickness)
                else:
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
                    
                # Store log event
                event_logs.append(f"Detected {name} at {int(confidence*100)}% conf.")

        return frame, stats, hardhat_detected, person_detected, event_logs, violation_crops
