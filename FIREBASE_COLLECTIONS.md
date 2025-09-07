# Firebase Firestore Collections - Safai Sathi

This document describes the Firebase Firestore collections used in the Safai Sathi garbage detection system.

## Project Configuration

- **Project ID:** `safai-saathi`
- **Database:** Firestore (NoSQL)
- **Authentication:** Firebase Admin SDK + REST API fallback

## Collections Overview

### 1. `detection_logs` Collection

**Purpose:** Stores individual detection events from laptop webcam in real-time.

**Source:** Laptop webcam (`file_path == "camera"`)

**Data Structure:**
```json
{
  "type": "individual_detection",
  "detection_count": 3,
  "confidence_scores": [0.85, 0.92, 0.78],
  "location": {
    "source": "GPS" | "IP",
    "latitude": 28.6139,
    "longitude": 77.2090,
    "accuracy": "±5 meters",
    "address": "New Delhi, India",
    "city": "New Delhi",
    "country": "India",
    "timestamp": "2024-01-15T10:30:45.123Z"
  },
  "timestamp": "2024-01-15T10:30:45.123Z",
  "saved_at": "2024-01-15T10:30:45.456Z",
  "source": "python_backend"
}
```

**Fields Description:**
- `type`: Always "individual_detection"
- `detection_count`: Number of garbage objects detected in the frame
- `confidence_scores`: Array of confidence values (0.0-1.0) for each detection
- `location`: Geolocation data (GPS preferred, IP fallback)
- `timestamp`: When the detection occurred
- `saved_at`: When the record was saved to Firebase
- `source`: Data source identifier

**Trigger:** Every 30 frames when using laptop webcam

**Code Location:** `app.py` line 375

---

### 2. `model_results` Collection

**Purpose:** Stores comprehensive detection reports and model analysis results.

**Source:** Both laptop webcam and ngrok stream (mobile CCTV)

**Data Structure:**
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "gps_location": {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "accuracy": "±5 meters",
    "address": "New Delhi, India",
    "source": "GPS",
    "timestamp": "2024-01-15T10:30:45.123Z"
  },
  "detection_summary": {
    "total_detections": 15,
    "average_confidence": 87.5,
    "max_confidence": 95.2,
    "min_confidence": 72.1,
    "overflow_score": 78.3,
    "detection_frequency": 5,
    "status": "HIGH_OVERFLOW" | "MEDIUM_OVERFLOW" | "LOW_OVERFLOW"
  },
  "recent_detections": [
    {
      "timestamp": "2024-01-15T10:30:45.123Z",
      "detection_count": 3,
      "confidence_scores": [85.2, 92.1, 78.5],
      "average_confidence": 85.3,
      "location": {
        "source": "GPS",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "city": "New Delhi",
        "country": "India",
        "address": "New Delhi, India"
      }
    }
  ],
  "saved_at": "2024-01-15T10:30:45.456Z",
  "source": "python_backend"
}
```

**Fields Description:**
- `timestamp`: Report generation time
- `gps_location`: Current GPS coordinates (if available)
- `detection_summary`: Aggregated statistics over last 30 seconds
  - `total_detections`: Total number of detections
  - `average_confidence`: Average confidence percentage
  - `max_confidence`: Highest confidence score
  - `min_confidence`: Lowest confidence score
  - `overflow_score`: Calculated overflow severity (0-100)
  - `detection_frequency`: Number of detection events
  - `status`: Overflow severity level
- `recent_detections`: Array of last 10 individual detections
- `saved_at`: When the report was saved to Firebase
- `source`: Data source identifier

**Trigger:** Every 30 seconds via periodic reports

**Code Location:** `app.py` line 241

---

## Data Flow

### Laptop Webcam Flow:
1. Camera captures frame → YOLO detection → Individual detection logged to `detection_logs`
2. Every 30 seconds → Comprehensive report generated → Saved to `model_results`

### Ngrok/Mobile Stream Flow:
1. Mobile CCTV stream → YOLO detection → Individual detection logged to `detection_logs`
2. Every 30 seconds → Comprehensive report generated → Saved to `model_results`

## Firebase Configuration

### Admin SDK Setup:
```python
# Initialize Firebase Admin SDK
firebase_admin.initialize_app()
db = firestore.client()
```

### REST API Fallback:
```python
# Fallback when Admin SDK fails
project_id = "safai-saathi"
api_key = "AIzaSyALVkB5jfl6O0CLNBtGmaX87Kc6UBu2TLE"
url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/{collection_name}"
```

### Frontend Configuration:
```javascript
// Firebase config
const firebaseConfig = {
  apiKey: "AIzaSyALVkB5jfl6O0CLNBtGmaX87Kc6UBu2TLE",
  authDomain: "safai-saathi.firebaseapp.com",
  projectId: "safai-saathi",
  storageBucket: "safai-saathi.firebasestorage.app",
  messagingSenderId: "6015045092",
  appId: "1:6015045092:web:a31cf2a86330fac60d4bf1",
  measurementId: "G-ZMM65PNXCL"
};
```

## Collection Usage Summary

| Collection | Primary Source | Data Type | Frequency | Purpose |
|------------|----------------|-----------|-----------|---------|
| `detection_logs` | Laptop Webcam | Individual detections | Every 30 frames | Real-time monitoring |
| `model_results` | Both sources | Comprehensive reports | Every 30 seconds | Analytics & reporting |

## Security Notes

- API keys are exposed in the code (should be moved to environment variables)
- Consider implementing proper authentication for production use
- Review Firestore security rules for data access control

## Monitoring & Analytics

The `model_results` collection is designed for:
- Dashboard analytics
- Historical trend analysis
- Overflow severity monitoring
- Location-based garbage detection mapping

The `detection_logs` collection is designed for:
- Real-time monitoring
- Individual detection tracking
- Performance analysis
- Debugging and troubleshooting

## API Endpoints

- `GET /location_logs` - Returns all detection logs
- `GET /json_report` - Returns current comprehensive report
- `GET /current_location` - Returns current GPS/IP location
- `POST /save_gps_location` - Saves GPS coordinates from mobile device
- `GET /gps_status` - Returns GPS availability status
