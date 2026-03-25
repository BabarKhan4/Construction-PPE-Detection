# Security Policy

## Supported Versions

Security updates and patches are actively maintained for the following milestone releases:

| Version | Supported          |
| ------- | ------------------ |
| v2.0.x  | :white_check_mark: |
| v1.0.x  | :x:                |

*(Note: v1 builds encompassing legacy `webcam.py` direct hardware bindings and SMTP multi-threading are formally deprecated and no longer receive security back-ports.)*

## Known Security Constraints

### Browser Hardware Protocols (WebRTC)
The system's `Live Webcam` component utilizes Web Real-Time Communication (WebRTC) arrays via `streamlit-webrtc`. Modern browsers strictly enforce HTTPS constraints for `getUserMedia` bindings. 
- If deployed across a Wide Area Network (WAN) or public cloud infrastructure, you **must** terminate the connection via TLS/SSL (HTTPS). The webcam frame will fail securely without triggering if exposed over standard HTTP.

### External Resource Ingestion
The `YouTube Stream Processor` utilizes `yt-dlp` to parse complex URLs and execute external requests.
- Input fields should ideally be sanitized by an external Reverse Proxy layer if hosting this application publicly to prevent Server Side Request Forgery (SSRF) leveraging the underlying `requests` dependencies. 

### Model Weight Provenance
The model weights (`Model/ppe.pt`) are loaded directly into the PyTorch runtime. PyTorch serialization utilizes `pickle` logic which is inherently insecure if reading unverified files.
- **Never** replace `ppe.pt` with community-sourced models without running an internal security audit or utilizing the `weights_only=True` PyTorch loading protocol where applicable.

## Reporting a Vulnerability

If you identify a vulnerability within the inference pipeline or the data handlers, please utilize standard internal channels. Do NOT open a public GitHub issue for Remote Code Execution (RCE) or Critical Data Exposure discoveries.

Include in your report:
- Type of vulnerability.
- Proof of Concept (PoC) code or instructions.
- Potential impact vectors on the underlying OS.
