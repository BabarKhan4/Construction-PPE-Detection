# GEAR Image Dataset (Internal)

## Overview
This proprietary dataset comprises highly curated bounding-box mappings utilized to directly train the core `ppe.pt` YOLOv8 classification model. The image matrices were rigorously audited by internal data integrity specialists, specifically targeting high-contrast edge cases, extreme weather occlusion, and unpredictable spatial orientations common across industrial construction sites.

## Classification Classes
The data exclusively maps to the following standard index hierarchy:
- `0`: Hardhat
- `1`: Mask
- `2`: NO-Hardhat
- `3`: NO-Mask
- `4`: NO-Safety Vest
- `5`: Person
- `6`: Safety Cone
- `7`: Safety Vest
- `8`: Machinery
- `9`: Vehicle

## Usage & Retention Policy
The contents of this image directory, encompassing all XML/YOLO algorithmic annotations and structural mappings, represent internal intellectual property.

- **Containerization:** Do NOT push this directory into cloud execution pods or external registries. All dataset directories are restricted via the root `.dockerignore` protocol.
- **Modification Protocol:** Ensure all visual pipeline augmentations respect the exact 10-class architectural mapping listed above to prevent inference degradation.

**CONFIDENTIAL:** For internal algorithmic retraining and edge-device compilation purposes only.
