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


# Flask APPLICATION
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Load the YOLOv8 model (using the trained weights)
model = YOLO('best.pt')

# Flag to indicate if the script should terminate
terminate_flag = False

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

# Function to generate and print JSON report
def generate_json_report():
    """Generate a comprehensive JSON report with all detection data"""
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
    total_detections = sum(log['detection_count'] for log in recent_logs)
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
            "confidence_scores": [round(score * 100, 2) for score in log['confidence_scores']],
            "average_confidence": round(sum(log['confidence_scores']) / len(log['confidence_scores']) * 100, 2) if log['confidence_scores'] else 0.0,
            "location": {
                "source": log['location'].get('source', 'IP'),
                "latitude": log['location'].get('latitude'),
                "longitude": log['location'].get('longitude'),
                "city": log['location'].get('city'),
                "country": log['location'].get('country'),
                "address": log['location'].get('address')
            }
        }
        report["recent_detections"].append(detection_entry)
    
    # Print the JSON report to console
    print("\n" + "="*80)
    print("🗑️  GARBAGE OVERFLOW DETECTION REPORT")
    print("="*80)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("="*80 + "\n")
    
    return report

# Function to run periodic JSON reports
def periodic_json_reports():
    """Run JSON report generation every 30 seconds"""
    while not terminate_flag:
        time.sleep(30)  # Wait 30 seconds
        if not terminate_flag:  # Check again after sleep
            generate_json_report()

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
        location_data = gps_location.copy()
        location_data['source'] = 'GPS'
    else:
        client_ip = get_client_ip()
        location_data = get_location_from_ip(client_ip)
    
    if location_data:
        log_entry = {
            'detection_count': detection_count,
            'confidence_scores': confidence_scores,
            'location': location_data,
            'timestamp': datetime.now().isoformat()
        }
        detection_logs.append(log_entry)
        
        # Keep only the last 100 logs to prevent memory issues
        if len(detection_logs) > 100:
            detection_logs.pop(0)

# Define a generator function to stream video frames to the web page
def generate(file_path):
    if file_path == "camera":
        cap = cv2.VideoCapture(0)
    elif file_path == "ngrok":
        # Use the ngrok URL for mobile phone CCTV
        # Try different possible video stream endpoints
        possible_urls = [
            "https://0be173065b6e.ngrok-free.app/video",
        ]
        
        cap = None
        for url in possible_urls:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                break
            else:
                cap.release()
        
        if not cap or not cap.isOpened():
            return
    else:
        cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        # Read a frame from the video file
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame (suppress output)
            results = model(frame, verbose=False)

            # Extract detection information
            detections = results[0].boxes
            if detections is not None and len(detections) > 0:
                detection_count = len(detections)
                confidence_scores = detections.conf.tolist() if detections.conf is not None else []
                
                # Log detection with geolocation (every 30 frames to avoid spam)
                if hasattr(generate, 'frame_count'):
                    generate.frame_count += 1
                else:
                    generate.frame_count = 1
                
                if generate.frame_count % 30 == 0:  # Log every 30 frames
                    log_detection_with_location(detection_count, confidence_scores)

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

# Route to get current JSON report
@app.route('/json_report')
def get_json_report():
    return jsonify(generate_json_report())

# Define a route to serve the HTML page with the file upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    global terminate_flag
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
        return render_template('index.html')

@app.route('/stop', methods=['POST'])
def stop():
    global terminate_flag
    terminate_flag = True
    return "Process has been Terminated"

if __name__ == '__main__':
    # Start the periodic JSON reporting thread
    report_thread = threading.Thread(target=periodic_json_reports, daemon=True)
    report_thread.start()
    
    print("🚀 Starting Smart Garbage Detection System...")
    print("📊 JSON reports will be generated every 30 seconds")
    print("🌐 Web interface available at: http://localhost:5000")
    print("📋 Manual JSON report available at: http://localhost:5000/json_report")
    print("="*60)
    
    # Suppress Flask and other library output
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
