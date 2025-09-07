import cv2
from ultralytics import YOLO
import argparse
from flask import Flask, Response, render_template, request, jsonify
import tempfile
import os
import requests
import json
from datetime import datetime
import threading
import time
import firebase_admin
from firebase_admin import credentials, firestore
import urllib.request
import urllib.parse
import numpy as np


# Flask APPLICATION
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Initialize Firebase Admin SDK
try:
    # Check if Firebase app is already initialized
    if not firebase_admin._apps:
        # Initialize Firebase Admin SDK with default credentials
        # For production, you should use a service account key file
        firebase_admin.initialize_app()
    
    db = firestore.client()
    print("✅ Firebase Admin SDK initialized successfully")
except Exception as e:
    print(f"⚠️  Firebase initialization failed: {e}")
    print("📝 Note: Terminal reports will not be saved to Firebase")
    print("💡 To fix this, you need to set up Firebase authentication:")
    print("   1. Go to Firebase Console > Project Settings > Service Accounts")
    print("   2. Generate a new private key")
    print("   3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    print("   4. Or place the service account key file in the project directory")
    db = None

# Load the YOLOv8 model (using the trained weights)
model = YOLO('best.pt')

# Flag to indicate if the script should terminate
terminate_flag = False

# Flag to track if video stream is active
stream_active = False

# Track last detection to prevent duplicates
last_detection = None

# Store detection data with location
detection_logs = []
gps_location = None  # Store the latest GPS location

# JSON report data
json_report_data = {
    "timestamp": None,
    "gps_location": None,
    "detection_summary": {
        "total_detections": 0,
        "average_confidence": 0.0,
        "max_confidence": 0.0,
        "min_confidence": 1.0,
        "overflow_score": 0.0
    },
    "recent_detections": []
}

# Function to save data to Firebase Firestore using REST API (fallback)
def save_to_firebase_rest(data, collection_name="model_results"):
    """Save data to Firebase using REST API as fallback"""
    try:
        # Firebase project configuration
        project_id = "safai-saathi"
        api_key = "AIzaSyALVkB5jfl6O0CLNBtGmaX87Kc6UBu2TLE"
        
        # Add timestamp to the data
        data_with_timestamp = {
            **data,
            "saved_at": datetime.now().isoformat(),
            "source": "python_backend_rest"
        }
        
        # Firebase REST API endpoint
        url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/{collection_name}"
        
        # Prepare the data for Firestore REST API format
        firestore_data = {
            "fields": {}
        }
        
        # Convert Python data to Firestore format
        def convert_to_firestore_value(value):
            if isinstance(value, str):
                return {"stringValue": value}
            elif isinstance(value, (int, float)):
                return {"doubleValue": value}
            elif isinstance(value, bool):
                return {"booleanValue": value}
            elif isinstance(value, list):
                # Handle arrays properly - convert each element recursively
                array_values = []
                for item in value:
                    if isinstance(item, dict):
                        # Convert dict items to mapValue
                        map_value = {"fields": {}}
                        for k, v in item.items():
                            map_value["fields"][k] = convert_to_firestore_value(v)
                        array_values.append({"mapValue": map_value})
                    else:
                        array_values.append(convert_to_firestore_value(item))
                return {"arrayValue": {"values": array_values}}
            elif isinstance(value, dict):
                # Convert nested dict to mapValue recursively
                map_value = {"fields": {}}
                for k, v in value.items():
                    map_value["fields"][k] = convert_to_firestore_value(v)
                return {"mapValue": map_value}
            else:
                return {"stringValue": str(value)}
        
        for key, value in data_with_timestamp.items():
            firestore_data["fields"][key] = convert_to_firestore_value(value)
        
        # Make the request
        req = urllib.request.Request(
            url,
            data=json.dumps(firestore_data).encode('utf-8'),
            headers={
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': api_key
            }
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            doc_id = result['name'].split('/')[-1]
            print(f"✅ Saved to Firebase Firestore (REST API) with ID: {doc_id}")
            return doc_id
            
    except Exception as e:
        print(f"❌ Error saving to Firebase via REST API: {e}")
        return None

# Function to save data to Firebase Firestore
def save_to_firebase(data, collection_name="model_results"):
    """Save data to Firebase Firestore"""
    if db is None:
        print("⚠️  Firebase Admin SDK not available, trying REST API fallback...")
        return save_to_firebase_rest(data, collection_name)
    
    try:
        # Add timestamp to the data
        data_with_timestamp = {
            **data,
            "saved_at": datetime.now().isoformat(),
            "source": "python_backend"
        }
        
        # Save to Firestore
        doc_ref = db.collection(collection_name).add(data_with_timestamp)
        doc_id = doc_ref[1].id
        print(f"✅ Saved to Firebase Firestore with ID: {doc_id}")
        return doc_id
    except Exception as e:
        print(f"❌ Error saving to Firebase Admin SDK: {e}")
        print("🔄 Trying REST API fallback...")
        return save_to_firebase_rest(data, collection_name)

# Function to generate and print JSON report
def generate_json_report():
    """Generate a comprehensive JSON report with all detection data - always returns JSON after 30 seconds"""
    global json_report_data, detection_logs, gps_location
    
    current_time = datetime.now()
    
    # Get recent detections (last 30 seconds worth)
    recent_logs = []
    if detection_logs:
        # Get logs from the last 30 seconds
        cutoff_time = current_time.timestamp() - 30
        recent_logs = [log for log in detection_logs 
                      if datetime.fromisoformat(log['timestamp']).timestamp() > cutoff_time]
    
    # Calculate detection statistics
    total_detections = sum(log['detection_count'] for log in recent_logs) if recent_logs else 0
    
    # Prepare GPS location data
    gps_data = None
    if gps_location:
        gps_data = {
            "latitude": gps_location.get('latitude'),
            "longitude": gps_location.get('longitude'),
            "accuracy": gps_location.get('accuracy'),
            "address": gps_location.get('address'),
            "source": gps_location.get('source', 'GPS'),
            "timestamp": gps_location.get('timestamp')
        }
    
    # Handle case when no garbage is detected
    if not recent_logs or total_detections == 0:
        # Create individual detection log entry for no-detection
        individual_no_detection = {
            "confidence_scores": [0, 0],  # Array of zeros for no detection
            "createdAt": current_time.isoformat(),
            "detection_count": 0,
            "location": gps_data,
            "source": "auto_save",
            "timestamp": current_time.isoformat(),
            "type": "detection_log"
        }
        
        # Create individual detection entry for no-detection (same format as regular detections)
        individual_detection_entry = {
            "timestamp": current_time.isoformat(),
            "detection_count": 0,
            "confidence_scores": [0.0, 0.0],  # Array of zeros for no detection
            "average_confidence": 0.0,
            "location": gps_data,
            "working_area": gps_data.get('address', 'Unknown').split(',')[0].strip() if gps_data and gps_data.get('address') else 'Unknown'
        }
        
        # Create periodic report structure for no-detection
        no_detection_report = {
            "timestamp": current_time.isoformat(),
            "gps_location": gps_data,
            "detection_summary": {
                "total_detections": 0,
                "average_confidence": 0.0,
                "max_confidence": 0.0,
                "min_confidence": 0.0,
                "overflow_score": 0.0,
                "detection_frequency": 0,
                "status": "NO_DETECTIONS"
            },
            "recent_detections": [individual_detection_entry]
        }
        
        # Print the no-detection report
        print("\n" + "="*80)
        print("🗑️  GARBAGE OVERFLOW DETECTION REPORT")
        print("📊 Total detections in last 30 seconds: 0")
        print("="*80)
        print(json.dumps(no_detection_report, indent=2, ensure_ascii=False))
        print("="*80 + "\n")
        
        # Save individual detection log to detection_logs collection
        print("💾 Saving individual no-detection log to detection_logs...")
        detection_doc_id = save_to_firebase(individual_no_detection, "detection_logs")
        if detection_doc_id:
            print(f"✅ Individual no-detection log saved to detection_logs with document ID: {detection_doc_id}")
        else:
            print("❌ Failed to save individual no-detection log to detection_logs")
        
        # Save periodic report to model_results collection
        print("💾 Saving no-detection periodic report to model_results...")
        model_doc_id = save_to_firebase(no_detection_report, "model_results")
        if model_doc_id:
            print(f"✅ No-detection periodic report saved to model_results with document ID: {model_doc_id}")
        else:
            print("❌ Failed to save no-detection periodic report to model_results")
        
        return no_detection_report
    
    # If garbage is detected, calculate full statistics
    all_confidences = []
    for log in recent_logs:
        all_confidences.extend(log['confidence_scores'])
    
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    max_confidence = max(all_confidences) if all_confidences else 0.0
    min_confidence = min(all_confidences) if all_confidences else 0.0
    
    # Calculate overflow score (based on detection frequency and confidence)
    overflow_score = min(100.0, (total_detections * avg_confidence * 10))
    
    # Prepare GPS location data
    gps_data = None
    if gps_location:
        gps_data = {
            "latitude": gps_location.get('latitude'),
            "longitude": gps_location.get('longitude'),
            "accuracy": gps_location.get('accuracy'),
            "address": gps_location.get('address'),
            "source": gps_location.get('source', 'GPS'),
            "timestamp": gps_location.get('timestamp')
        }
    
    # Generate the report
    report = {
        "timestamp": current_time.isoformat(),
        "gps_location": gps_data,
        "detection_summary": {
            "total_detections": total_detections,
            "average_confidence": round(avg_confidence * 100, 2),
            "max_confidence": round(max_confidence * 100, 2),
            "min_confidence": round(min_confidence * 100, 2),
            "overflow_score": round(overflow_score, 2),
            "detection_frequency": len(recent_logs),
            "status": "HIGH_OVERFLOW" if overflow_score > 70 else "MEDIUM_OVERFLOW" if overflow_score > 30 else "LOW_OVERFLOW"
        },
        "recent_detections": []
    }
    
    # Add recent detection details
    for log in recent_logs[-10:]:  # Last 10 detections
        detection_entry = {
            "timestamp": log['timestamp'],
            "detection_count": log['detection_count'],
            "confidence_scores": [round(score, 4) for score in log['confidence_scores']],
            "average_confidence": round(sum(log['confidence_scores']) / len(log['confidence_scores']), 4) if log['confidence_scores'] else 0.0,
            "location": {
                "source": log['location'].get('source', 'IP'),
                "latitude": log['location'].get('latitude'),
                "longitude": log['location'].get('longitude'),
                "accuracy": log['location'].get('accuracy'),
                "address": log['location'].get('address'),
                "timestamp": log['location'].get('timestamp')
            },
            "working_area": log.get('working_area', 'Unknown')
        }
        report["recent_detections"].append(detection_entry)
    
    # Print the JSON report to console
    print("\n" + "="*80)
    print("🗑️  GARBAGE OVERFLOW DETECTION REPORT")
    print(f"📊 Total detections in last 30 seconds: {total_detections}")
    print("="*80)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("="*80 + "\n")
    
    # Save the report to Firebase Firestore
    print("💾 Saving report to Firebase Firestore...")
    firebase_doc_id = save_to_firebase(report)
    if firebase_doc_id:
        print(f"✅ Report saved to Firebase with document ID: {firebase_doc_id}")
    else:
        print("❌ Failed to save report to Firebase")
    
    return report

# Function to run periodic JSON reports
def periodic_json_reports():
    """Run JSON report generation every 30 seconds - only when stream is active"""
    while not terminate_flag:
        time.sleep(30)  # Wait 30 seconds
        if not terminate_flag and stream_active:  # Only generate reports when stream is active
            report = generate_json_report()
            if report["detection_summary"]["total_detections"] == 0:
                print("⏱️  30-second interval completed - no garbage detected, returning zero report")
            else:
                print("✅ Garbage detection report generated successfully")
        elif not stream_active:
            print("📹 Video stream not active - skipping periodic report")

# Geolocation function using a free IP geolocation service
def get_location_from_ip(ip_address):
    try:
        # If we have a localhost IP, try to get public IP first
        if ip_address in ['127.0.0.1', 'localhost', '::1']:
                try:
                    response = requests.get('https://api.ipify.org?format=json', timeout=5)
                    if response.status_code == 200:
                        ip_address = response.json().get('ip')
                except:
                    # Try alternative service
                    try:
                        response = requests.get('https://httpbin.org/ip', timeout=5)
                        if response.status_code == 200:
                            ip_address = response.json().get('origin', '').split(',')[0].strip()
                    except:
                        pass
        
        # Now get location data
        response = requests.get(f'https://ipapi.co/{ip_address}/json/', timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'source': 'IP',
                'ip': ip_address,
                'city': data.get('city', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'country': data.get('country_name', 'Unknown'),
                'latitude': data.get('latitude'),
                'longitude': data.get('longitude'),
                'accuracy': 'City-level (~10km)',
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        pass
    
    return None

# Reverse geocoding function to get address from coordinates
def get_address_from_coords(lat, lng):
    try:
        # Using Nominatim (OpenStreetMap) for reverse geocoding
        response = requests.get(
            f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}',
            headers={'User-Agent': 'GarbageDetectionApp/1.0'},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get('display_name', 'Address not found')
    except Exception as e:
        pass
    return None

# Get client IP address
def get_client_ip():
    # Check various headers for real IP
    possible_headers = [
        'HTTP_X_FORWARDED_FOR',
        'HTTP_X_REAL_IP',
        'HTTP_X_FORWARDED',
        'HTTP_X_CLUSTER_CLIENT_IP',
        'HTTP_FORWARDED_FOR',
        'HTTP_FORWARDED',
        'REMOTE_ADDR'
    ]
    
    for header in possible_headers:
        ip = request.environ.get(header)
        if ip and ip != '127.0.0.1' and ip != 'localhost':
            # Handle comma-separated IPs (X-Forwarded-For can have multiple IPs)
            if ',' in ip:
                ip = ip.split(',')[0].strip()
            return ip
    
    # If we still have localhost, try to get public IP
    try:
        # Use a service to get the public IP
        response = requests.get('https://httpbin.org/ip', timeout=5)
        if response.status_code == 200:
            public_ip = response.json().get('origin', '').split(',')[0].strip()
            if public_ip and public_ip != '127.0.0.1':
                return public_ip
    except:
        pass
    
    # Fallback to environment IP
    return request.environ.get('REMOTE_ADDR', '127.0.0.1')

# Store detection data with location
def log_detection_with_location(detection_count, confidence_scores):
    # Try to use GPS location first, fallback to IP location
    location_data = None
    
    if gps_location:
        location_data = {
            'accuracy': gps_location.get('accuracy', 'Unknown'),
            'address': gps_location.get('address'),
            'latitude': gps_location.get('latitude'),
            'longitude': gps_location.get('longitude'),
            'source': 'GPS',
            'timestamp': gps_location.get('timestamp')
        }
    else:
        client_ip = get_client_ip()
        ip_location = get_location_from_ip(client_ip)
        if ip_location:
            location_data = {
                'accuracy': ip_location.get('accuracy', 'City-level (~10km)'),
                'address': f"{ip_location.get('city', 'Unknown')}, {ip_location.get('region', 'Unknown')}, {ip_location.get('country', 'Unknown')}",
                'latitude': ip_location.get('latitude'),
                'longitude': ip_location.get('longitude'),
                'source': 'IP',
                'timestamp': ip_location.get('timestamp')
            }
    
    if location_data:
        # Extract working area from address (first part before comma)
        working_area = "Unknown"
        if location_data.get('address'):
            working_area = location_data['address'].split(',')[0].strip()
        
        # Create the exact structure you specified
        detection_data = {
            'assignedAt': datetime.now().isoformat() + 'Z',
            'confidence_scores': confidence_scores,
            'createdAt': datetime.now(),  # Firebase will convert this to timestamp
            'detection_count': detection_count,
            'location': location_data,
            'source': 'auto_save',
            'staffId': 'aD5OR3uuKciXSgOSHvcn',
            'timestamp': datetime.now().isoformat(),
            'type': 'detection_log',
            'updatedAt': datetime.now().isoformat() + 'Z',
            'workStatus': 'in_progress',
            'working_area': working_area
        }
        
        # Store in local logs with simplified structure for periodic reports
        log_entry = {
            'detection_count': detection_count,
            'confidence_scores': confidence_scores,
            'location': location_data,
            'timestamp': datetime.now().isoformat(),
            'working_area': working_area
        }
        detection_logs.append(log_entry)
        
        # Save to Firebase with the exact structure you want
        save_to_firebase(detection_data, "detection_logs")
        
        # Keep only the last 100 logs to prevent memory issues
        if len(detection_logs) > 100:
            detection_logs.pop(0)

# Define a generator function to stream video frames to the web page
def generate(file_path):
    global stream_active
    
    if file_path == "camera":
        cap = cv2.VideoCapture(0)
    elif file_path == "ngrok":
        # Use the ngrok URL for mobile phone CCTV
        # Try different possible video stream endpoints
        possible_urls = [
            "https://b35ac7b4e14e.ngrok-free.app/video",
            "http://localhost:8080/video",  # Local fallback
            "http://192.168.1.100:8080/video",  # Common local network IP
        ]
        
        cap = None
        successful_url = None
        for url in possible_urls:
            print(f"🔄 Trying to connect to: {url}")
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    successful_url = url
                    print(f"✅ Successfully connected to: {url}")
                    break
                else:
                    cap.release()
                    cap = None
            else:
                if cap:
                    cap.release()
                cap = None
        
        if not cap or not cap.isOpened():
            print("❌ Failed to connect to any mobile CCTV stream")
            print("📱 Mobile CCTV Setup Instructions:")
            print("   1. Install a mobile camera streaming app (e.g., 'IP Webcam' for Android)")
            print("   2. Start the app and note the streaming URL")
            print("   3. Update the ngrok URL in the code or use the local network URL")
            print("   4. Make sure your mobile device and computer are on the same network")
            stream_active = False
            return
    else:
        cap = cv2.VideoCapture(file_path)
    
    # Set stream as active when we successfully open a video source
    if cap and cap.isOpened():
        stream_active = True
        last_detection = None  # Reset detection tracking for new stream
        print("📹 Video stream started - periodic reports will now be generated")
    while cap.isOpened():
        # Read a frame from the video file
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame (suppress output)
            results = model(frame, verbose=False)

            # Extract detection information
            detections = results[0].boxes
            
            # Initialize frame counter
            if hasattr(generate, 'frame_count'):
                generate.frame_count += 1
            else:
                generate.frame_count = 1
            
            # Log detection with geolocation (every 30 frames to avoid spam)
            if generate.frame_count % 30 == 0:  # Log every 30 frames
                if detections is not None and len(detections) > 0:
                    # Garbage detected
                    detection_count = len(detections)
                    confidence_scores = detections.conf.tolist() if detections.conf is not None else []
                else:
                    # No garbage detected - log zero detection
                    detection_count = 0
                    confidence_scores = [0.0, 0.0]  # Array of zeros for no detection
                
                # Check if this detection is different from the last one
                current_detection = {
                    'detection_count': detection_count,
                    'confidence_scores': confidence_scores
                }
                
                # Only log if detection is different from last detection
                if last_detection != current_detection:
                    log_detection_with_location(detection_count, confidence_scores)
                    last_detection = current_detection
                    print(f"🔄 New detection logged: {detection_count} objects, confidence: {confidence_scores}")
                else:
                    print(f"⏭️  Skipping duplicate detection: {detection_count} objects, confidence: {confidence_scores}")

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)

            # Yield the JPEG data to Flask
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            if cv2.waitKey(1) == 27 or terminate_flag:  # Exit when ESC key is pressed or terminate flag is set
                break
        else:
            # Break the loop if the video file capture fails
            break
    
    # Set stream as inactive when video stream ends
    stream_active = False
    print("📹 Video stream stopped - periodic reports will be paused")
    cap.release()
    os._exit(0)  # Terminate the script when the video stream ends or terminate flag is set

# Define a route to serve the video stream
@app.route('/video_feed')
def video_feed():
    file_path = request.args.get('file')
    return Response(generate(file_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to get location logs
@app.route('/location_logs')
def location_logs():
    return jsonify(detection_logs)

# Route to get current location
@app.route('/current_location')
def current_location():
    # Prefer GPS location if available
    if gps_location:
        return jsonify(gps_location)
    
    # Fallback to IP location
    client_ip = get_client_ip()
    location_data = get_location_from_ip(client_ip)
    return jsonify(location_data) if location_data else jsonify({"error": "Location not found"})

# Route to save GPS location from mobile device
@app.route('/save_gps_location', methods=['POST'])
def save_gps_location():
    global gps_location
    try:
        data = request.get_json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        accuracy = data.get('accuracy')
        timestamp = data.get('timestamp')
        
        if latitude and longitude:
            # Get address from coordinates
            address = get_address_from_coords(latitude, longitude)
            
            gps_location = {
                'source': 'GPS',
                'latitude': latitude,
                'longitude': longitude,
                'accuracy': f"±{accuracy:.0f} meters" if accuracy else "Unknown",
                'address': address,
                'timestamp': timestamp
            }
            
            return jsonify({
                'status': 'success',
                'message': 'GPS location saved',
                'address': address
            })
        else:
            return jsonify({'error': 'Invalid coordinates'}), 400
            
    except Exception as e:
        return jsonify({'error': 'Failed to save location'}), 500

# Route to get GPS status
@app.route('/gps_status')
def gps_status():
    return jsonify({
        'has_gps': gps_location is not None,
        'location': gps_location
    })

# Route to test mobile CCTV connection
@app.route('/test_mobile_cctv')
def test_mobile_cctv():
    """Test connection to mobile CCTV stream"""
    possible_urls = [
        "https://932a43845154.ngrok-free.app/video",
        "http://localhost:8080/video",
        "http://192.168.1.100:8080/video",
    ]
    
    results = []
    for url in possible_urls:
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    results.append({
                        'url': url,
                        'status': 'success',
                        'message': 'Connection successful'
                    })
                else:
                    results.append({
                        'url': url,
                        'status': 'failed',
                        'message': 'Cannot read frames'
                    })
                cap.release()
            else:
                results.append({
                    'url': url,
                    'status': 'failed',
                    'message': 'Cannot open connection'
                })
        except Exception as e:
            results.append({
                'url': url,
                'status': 'error',
                'message': str(e)
            })
    
    return jsonify({
        'results': results,
        'setup_instructions': [
            "1. Install 'IP Webcam' app on your Android device",
            "2. Start the app and go to 'Start server'",
            "3. Note the IP address shown (e.g., 192.168.1.100:8080)",
            "4. The video stream will be available at http://[IP]:8080/video",
            "5. Make sure both devices are on the same WiFi network",
            "6. For external access, use ngrok or similar tunneling service"
        ]
    })

# Route to get current JSON report
@app.route('/json_report')
def get_json_report():
    """API endpoint to get current JSON report - only when stream is active"""
    if not stream_active:
        return jsonify({"error": "Video stream not active", "message": "Start a video stream to generate reports"})
    report = generate_json_report()
    return jsonify(report)

# Route to fetch images from Firestore civilian collection
@app.route('/fetch_civilian_images')
def fetch_civilian_images():
    """Fetch all imageUrl from Firestore civilian collection using REST API"""
    try:
        # Use REST API to fetch civilian images
        project_id = "safai-saathi"
        api_key = "AIzaSyALVkB5jfl6O0CLNBtGmaX87Kc6UBu2TLE"
        
        # Firebase REST API endpoint for civilian collection
        url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/civilian"
        
        # Make the request
        req = urllib.request.Request(
            url,
            headers={
                'X-Goog-Api-Key': api_key
            }
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            
            images = []
            if 'documents' in result:
                for doc in result['documents']:
                    doc_id = doc['name'].split('/')[-1]
                    fields = doc.get('fields', {})
                    
                    # Extract imageUrl if it exists
                    if 'imageUrl' in fields:
                        image_url = fields['imageUrl'].get('stringValue', '')
                        
                        # Extract other fields
                        description = fields.get('description', {}).get('stringValue', '')
                        timestamp = fields.get('timestamp', {}).get('stringValue', '')
                        
                        # Extract location data
                        location = {}
                        if 'location' in fields:
                            location_fields = fields['location'].get('mapValue', {}).get('fields', {})
                            location = {
                                'accuracy': location_fields.get('accuracy', {}).get('doubleValue', 0),
                                'city': location_fields.get('city', {}).get('stringValue', ''),
                                'country': location_fields.get('country', {}).get('stringValue', ''),
                                'latitude': location_fields.get('latitude', {}).get('doubleValue', 0),
                                'longitude': location_fields.get('longitude', {}).get('doubleValue', 0)
                            }
                        
                        images.append({
                            'id': doc_id,
                            'imageUrl': image_url,
                            'timestamp': timestamp,
                            'location': location,
                            'description': description
                        })
            
            return jsonify({
                'success': True,
                'images': images,
                'count': len(images),
                'source': 'firestore_rest_api'
            })
        
    except Exception as e:
        print(f"❌ Error fetching civilian images via REST API: {e}")
        # Return empty result instead of error
        return jsonify({
            'success': True,
            'images': [],
            'count': 0,
            'message': f'Error fetching civilian images: {str(e)}'
        })

# Route to fetch images directly from Cloudinary (for testing)
@app.route('/fetch_cloudinary_images')
def fetch_cloudinary_images():
    """Fetch sample images from Cloudinary for testing"""
    try:
        # Sample Cloudinary URLs for testing
        sample_images = [
            {
                'id': 'cloudinary_1',
                'imageUrl': 'https://res.cloudinary.com/dq2oyfftd/image/upload/v1757212466/safai-citizen/blob_livahy.jpg',
                'timestamp': datetime.now().isoformat(),
                'location': {'city': 'Test City', 'country': 'Test Country'},
                'description': 'Sample garbage detection image 1'
            },
            {
                'id': 'cloudinary_2', 
                'imageUrl': 'https://res.cloudinary.com/dq2oyfftd/image/upload/v1757212466/safai-citizen/blob_livahy.jpg',
                'timestamp': datetime.now().isoformat(),
                'location': {'city': 'Test City', 'country': 'Test Country'},
                'description': 'Sample garbage detection image 2'
            },
            {
                'id': 'cloudinary_3',
                'imageUrl': 'https://res.cloudinary.com/dq2oyfftd/image/upload/v1757212466/safai-citizen/blob_livahy.jpg', 
                'timestamp': datetime.now().isoformat(),
                'location': {'city': 'Test City', 'country': 'Test Country'},
                'description': 'Sample garbage detection image 3'
            }
        ]
        
        return jsonify({
            'success': True,
            'images': sample_images,
            'count': len(sample_images),
            'source': 'cloudinary_samples'
        })
        
    except Exception as e:
        print(f"❌ Error fetching Cloudinary images: {e}")
        return jsonify({"error": str(e)}), 500

# Route to analyze a specific image
@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Analyze an image from URL using YOLO model"""
    try:
        data = request.get_json()
        image_url = data.get('imageUrl')
        image_id = data.get('imageId')
        
        if not image_url:
            return jsonify({"error": "No imageUrl provided"}), 400
        
        # Download image from URL
        import urllib.request
        import numpy as np
        
        # Download the image
        with urllib.request.urlopen(image_url) as response:
            image_data = response.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        # Run YOLO inference
        results = model(image, verbose=False)
        
        # Extract detection information
        detections = results[0].boxes
        detection_count = 0
        confidence_scores = []
        detection_details = []
        
        if detections is not None and len(detections) > 0:
            detection_count = len(detections)
            confidence_scores = detections.conf.tolist() if detections.conf is not None else []
            
            # Get bounding boxes and classes
            boxes = detections.xyxy.tolist() if detections.xyxy is not None else []
            classes = detections.cls.tolist() if detections.cls is not None else []
            
            # Create detailed detection information
            for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidence_scores)):
                detection_details.append({
                    'id': i + 1,
                    'class': int(cls),
                    'class_name': model.names[int(cls)] if int(cls) in model.names else 'unknown',
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3])
                    }
                })
        
        # Calculate statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        max_confidence = max(confidence_scores) if confidence_scores else 0.0
        min_confidence = min(confidence_scores) if confidence_scores else 0.0
        
        # Create analysis result
        analysis_result = {
            'imageId': image_id,
            'imageUrl': image_url,
            'timestamp': datetime.now().isoformat(),
            'detection_count': detection_count,
            'confidence_scores': [round(score, 3) for score in confidence_scores],
            'average_confidence': round(avg_confidence, 3),
            'max_confidence': round(max_confidence, 3),
            'min_confidence': round(min_confidence, 3),
            'detection_details': detection_details,
            'status': 'HIGH_OVERFLOW' if detection_count > 5 else 'MEDIUM_OVERFLOW' if detection_count > 2 else 'LOW_OVERFLOW' if detection_count > 0 else 'CLEAN'
        }
        
        # Save analysis result to Firebase in a new collection
        try:
            save_to_firebase(analysis_result, "cloudinary_analysis_results")
            analysis_result['saved_to_firebase'] = True
        except Exception as e:
            print(f"⚠️ Could not save to Firebase: {e}")
            analysis_result['saved_to_firebase'] = False
            analysis_result['firebase_error'] = str(e)
        
        return jsonify({
            'success': True,
            'analysis': analysis_result
        })
        
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")
        return jsonify({"error": str(e)}), 500

# Route to generate comprehensive analysis report
@app.route('/generate_analysis_report')
def generate_analysis_report():
    """Generate a comprehensive report of all analyzed images"""
    try:
        if db is None:
            return jsonify({
                "success": True,
                "report": {
                    "report_timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total_images_analyzed": 0,
                        "total_detections": 0,
                        "average_detections_per_image": 0,
                        "average_confidence": 0,
                        "max_confidence": 0,
                        "min_confidence": 0,
                        "status_distribution": {'CLEAN': 0, 'LOW_OVERFLOW': 0, 'MEDIUM_OVERFLOW': 0, 'HIGH_OVERFLOW': 0}
                    },
                    "detailed_analyses": [],
                    "recommendations": ["Firebase not initialized - no analysis data available"],
                    "message": "Firebase not initialized"
                }
            })
        
        # Query the cloudinary_analysis_results collection
        results_ref = db.collection('cloudinary_analysis_results')
        docs = results_ref.stream()
        
        analyses = []
        total_detections = 0
        total_images = 0
        confidence_scores = []
        status_counts = {'CLEAN': 0, 'LOW_OVERFLOW': 0, 'MEDIUM_OVERFLOW': 0, 'HIGH_OVERFLOW': 0}
        
        for doc in docs:
            doc_data = doc.to_dict()
            analyses.append({
                'id': doc.id,
                'imageUrl': doc_data.get('imageUrl', ''),
                'timestamp': doc_data.get('timestamp', ''),
                'detection_count': doc_data.get('detection_count', 0),
                'average_confidence': doc_data.get('average_confidence', 0),
                'status': doc_data.get('status', 'UNKNOWN'),
                'detection_details': doc_data.get('detection_details', [])
            })
            
            total_images += 1
            total_detections += doc_data.get('detection_count', 0)
            confidence_scores.append(doc_data.get('average_confidence', 0))
            
            status = doc_data.get('status', 'UNKNOWN')
            if status in status_counts:
                status_counts[status] += 1
        
        # Calculate statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        max_confidence = max(confidence_scores) if confidence_scores else 0
        min_confidence = min(confidence_scores) if confidence_scores else 0
        
        # Generate comprehensive report
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_images_analyzed': total_images,
                'total_detections': total_detections,
                'average_detections_per_image': round(total_detections / total_images, 2) if total_images > 0 else 0,
                'average_confidence': round(avg_confidence, 3),
                'max_confidence': round(max_confidence, 3),
                'min_confidence': round(min_confidence, 3),
                'status_distribution': status_counts
            },
            'detailed_analyses': analyses,
            'recommendations': generate_recommendations(status_counts, total_images)
        }
        
        # Save the comprehensive report to Firebase
        save_to_firebase(report, "comprehensive_analysis_reports")
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        print(f"❌ Error generating analysis report: {e}")
        return jsonify({"error": str(e)}), 500

def generate_recommendations(status_counts, total_images):
    """Generate recommendations based on analysis results"""
    recommendations = []
    
    if status_counts['HIGH_OVERFLOW'] > total_images * 0.3:
        recommendations.append("🚨 High priority: Multiple areas show high garbage overflow. Immediate cleanup required.")
    
    if status_counts['MEDIUM_OVERFLOW'] > total_images * 0.4:
        recommendations.append("⚠️ Medium priority: Several areas need attention. Schedule cleanup within 24 hours.")
    
    if status_counts['LOW_OVERFLOW'] > total_images * 0.5:
        recommendations.append("📋 Low priority: Some areas show minor garbage accumulation. Regular maintenance recommended.")
    
    if status_counts['CLEAN'] > total_images * 0.7:
        recommendations.append("✅ Good: Most areas are clean. Continue current maintenance schedule.")
    
    if not recommendations:
        recommendations.append("📊 Analysis complete. Review individual results for specific recommendations.")
    
    return recommendations

# Define a route to serve the HTML page with the file upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    global terminate_flag, stream_active
    if request.method == 'POST':
        if request.form.get("camera") == "true":
            file_path = "camera"
        elif request.form.get("ngrok") == "true":
            file_path = "ngrok"
        elif 'file' in request.files:
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
        else:
            file_path = None
        return render_template('index.html', file_path=file_path)
    else:
        terminate_flag = False
        stream_active = False
        return render_template('index.html')

@app.route('/stop', methods=['POST'])
def stop():
    global terminate_flag, stream_active
    terminate_flag = True
    stream_active = False
    print("🛑 Process terminated - all reporting stopped")
    return "Process has been Terminated"

if __name__ == '__main__':
    # Start the periodic JSON reporting thread
    report_thread = threading.Thread(target=periodic_json_reports, daemon=True)
    report_thread.start()
    
    print("🚀 Starting Smart Garbage Detection System...")
    print("📊 JSON reports will be generated every 30 seconds (with or without detections)")
    print("🌐 Web interface available at: http://localhost:5000")
    print("📋 Manual JSON report available at: http://localhost:5000/json_report")
    print("="*60)
    
    # Suppress Flask and other library output
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
