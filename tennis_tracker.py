import os
import json
import cv2
import math
import time  
import numpy as np
import datetime
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


try:
    if not firebase_admin._apps:
        # Point to your secure private key file downloaded from Firebase Console
        cred = credentials.Certificate("firebase_credentials.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'YOUR Database URL here'
        })
        print("☁️ Firebase Connected Securely!")
except Exception as e:
    print(f"⚠️ Firebase initialization failed. Error: {e}")

# Change this to your exact Tailscale Pi address when streaming live on the court:
video_path = "http://XXX.XXX.XXX.XXX:XXX/XXX"
config_path = "court_config.json"

# Load YOLO Model (Medium model)
print("🧠 Loading YOLOv8 Model...")
model = YOLO('yolov8m.pt') 

print("⏳ Waiting for Raspberry Pi stream to come online...")
cap = cv2.VideoCapture(video_path)

while not cap.isOpened():
    print("⚠️ Stream not found. Retrying in 5 seconds... (Plug in your Pi at the court!)")
    time.sleep(5)
    cap = cv2.VideoCapture(video_path)

print("⚡ Connected to Pi Stream!")

video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps == 0:
    video_fps = 30.0
    
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
if frame_height == 0:
    frame_height = 720 

clicked_points = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 255, 255), -1)
        cv2.imshow("Calibrate Court", param)

ret, first_frame = cap.read()
if ret:
    if os.path.exists(config_path):
        print("💾 Loading court configuration profile...")
        with open(config_path, 'r') as f:
            saved = json.load(f)
            clicked_points = saved["corners"]
    else:
        clone = first_frame.copy()
        cv2.imshow("Calibrate Court", clone)
        cv2.setMouseCallback("Calibrate Court", click_event, clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(clicked_points) == 4:
            config_data = {"corners": clicked_points}
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            print("💾 Config Saved!")
        else:
            print("❌ Must click 4 points to calibrate.")
            exit()


lk_params = dict(
    winSize=(31, 31), 
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

tracking_points = np.array(clicked_points, dtype=np.float32).reshape(-1, 1, 2)
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Court Dimensions in cm
full_court_width = 823.0
full_court_length = 640.0
center_line_x = 411.5

flat_list_coordinates = [
    0.0, 0.0,
    full_court_width, 0.0,
    full_court_width, full_court_length,
    0.0, full_court_length
]
birds_eye_corners = np.array(flat_list_coordinates, dtype=np.float32).reshape(4, 2)


history_raw_pixels = []
history_flat_coordinates = []
history_matrices = []

last_known_px = None
consecutive_misses = 0
frame_count = 0


show_display_overlay = False
display_countdown = 0
display_bx_pixel, display_by_pixel = 0, 0
display_target_box, display_serve_result, display_serve_placement = "", "", ""

print("🚀 Starting Unified Live Court Tracking & Tracking Engine...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_pts, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, tracking_points, None, **lk_params
    )
    
    if new_pts is not None and len(new_pts) == 4 and np.all(st == 1):
        current_corners = new_pts.reshape(4, 2)
        tracking_points = new_pts.copy()
    else:
        current_corners = tracking_points.reshape(4, 2)
        
    frame_matrix = cv2.getPerspectiveTransform(current_corners, birds_eye_corners)
    old_gray = frame_gray.copy()

    
    [results] = model(frame, verbose=False)
    valid_balls = []
    best_ball = None 
    
    for box in results.boxes:
        name = model.names[int(box.cls)]
        if name != "sports ball":
            continue
            
        b_box = box.xyxy.cpu().numpy().flatten()
        xmin, ymin, xmax, ymax = float(b_box[0]), float(b_box[1]), float(b_box[2]), float(b_box[3])
        
        if (xmax - xmin) > 60 or (ymax - ymin) > 60:
            continue
            
        px_x = float((xmin + xmax) / 2.0)
        px_y = float(ymax) 
        
        valid_balls.append({"px_x": px_x, "px_y": px_y})

    if valid_balls:
        if last_known_px is None:
            valid_balls.sort(key=lambda b: b["px_y"])
            highest_ball = valid_balls[0]
            if highest_ball["px_y"] < (frame_height * 0.75):
                best_ball = highest_ball
        else:
            valid_balls.sort(key=lambda b: math.hypot(b["px_x"] - last_known_px[0], b["px_y"] - last_known_px[1]))
            dist = math.hypot(valid_balls[0]["px_x"] - last_known_px[0], valid_balls[0]["px_y"] - last_known_px[1])
            if dist < 250.0: 
                best_ball = valid_balls[0]

  
    if best_ball is not None:
        px_x = best_ball["px_x"]
        px_y = best_ball["px_y"]
        
        is_rolling = False
        recent_window = history_raw_pixels[-20:]
        recent_pts = [p for p in recent_window if p is not None]
        
        if len(recent_pts) >= 15:
            start_px = recent_pts[0]
            dist_covered = math.hypot(px_x - start_px[0], px_y - start_px[1])
            if dist_covered < 90.0:  
                is_rolling = True
                
        if is_rolling:
            best_ball = None       
            last_known_px = None   
            consecutive_misses = 20 

   
    if best_ball is not None:
        px_x = best_ball["px_x"]
        px_y = best_ball["px_y"]
        last_known_px = (px_x, px_y)
        consecutive_misses = 0
        
        point_mat = np.array([[[px_x, px_y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_mat, frame_matrix).flatten()
        flat_x = float(transformed[0])
        flat_y = float(transformed[1])
        
        history_raw_pixels.append((px_x, px_y))
        history_flat_coordinates.append((flat_x, flat_y))
    else:
        consecutive_misses += 1
        if consecutive_misses > 15:
            last_known_px = None 
            
        history_raw_pixels.append(None)
        history_flat_coordinates.append(None)

    history_matrices.append(frame_matrix)


    if consecutive_misses == 16:
        valid_indices = [i for i, pt in enumerate(history_flat_coordinates) if pt is not None]

        if len(valid_indices) >= 3:
            segments = []
            current_segment = []
            for i in range(len(valid_indices)):
                if not current_segment:
                    current_segment.append(valid_indices[i])
                else:
                    if valid_indices[i] - current_segment[-1] <= 15:
                        current_segment.append(valid_indices[i])
                    else:
                        segments.append(current_segment)
                        current_segment = [valid_indices[i]]
            if current_segment:
                segments.append(current_segment)

            if segments:
                best_segment = max(segments, key=len)

                if len(best_segment) >= 3:
                    start_f = best_segment[0]
                    end_f = best_segment[-1]

                    segment_frames = np.arange(start_f, end_f + 1)
                    seg_raw_xs = [float(history_raw_pixels[i][0]) for i in best_segment]
                    seg_raw_ys = [float(history_raw_pixels[i][1]) for i in best_segment]

                    interp_x = np.interp(segment_frames, best_segment, seg_raw_xs)
                    interp_y = np.interp(segment_frames, best_segment, seg_raw_ys)

                    candidate_indices = []
                    for i in range(2, len(interp_y) - 2):
                        if interp_y[i] == np.max(interp_y[i-2:i+3]):
                            f = segment_frames[i]
                            bx = int(interp_x[i])
                            by = int(interp_y[i])
                            flat_pt = cv2.perspectiveTransform(
                                np.array([[[float(bx), float(by)]]], dtype=np.float32), 
                                history_matrices[f]
                            ).flatten()
                            
                            if -300 < flat_pt[0] < (full_court_width + 300) and -300 < flat_pt[1] < (full_court_length + 300):
                                candidate_indices.append(i)

                    bounce_frame = None
                    if candidate_indices:
                        bounce_frame = segment_frames[candidate_indices[0]]

                    if bounce_frame is None:
                        best_y = -1
                        best_f = segment_frames[0]
                        
                        for i, f in enumerate(segment_frames):
                            bx = int(interp_x[i])
                            by = int(interp_y[i])
                            flat_pt = cv2.perspectiveTransform(
                                np.array([[[float(bx), float(by)]]], dtype=np.float32), 
                                history_matrices[f]
                            ).flatten()
                            
                            if -300 < flat_pt[0] < (full_court_width + 300) and -300 < flat_pt[1] < (full_court_length + 300):
                                if by > best_y:
                                    best_y = by
                                    best_f = f
                                if best_y - by > 15:
                                    break
                                    
                        bounce_frame = best_f

                    idx = bounce_frame - start_f
                    bx_pixel = int(interp_x[idx])
                    by_pixel = int(interp_y[idx])

                    bounce_flat_x, bounce_flat_y = cv2.perspectiveTransform(
                        np.array([[[float(bx_pixel), float(by_pixel)]]], dtype=np.float32), 
                        history_matrices[bounce_frame]
                    ).flatten()

                
                    total_dx_cm = 0.0
                    for i in range(1, len(best_segment)):
                        f1 = best_segment[i-1]
                        f2 = best_segment[i]
                        if f2 > bounce_frame:
                            break
                        total_dx_cm += (history_flat_coordinates[f2][0] - history_flat_coordinates[f1][0])

                    target_box = "DEUCE" if total_dx_cm > 0.0 else "AD"
                    actual_box = "AD" if bounce_flat_x < center_line_x else "DEUCE"

                    dist = abs(bounce_flat_x - center_line_x)
                    if dist < 137.1:
                        serve_placement = "Down the T"
                    elif dist > 274.3:
                        serve_placement = "Wide"
                    else:
                        serve_placement = "Middle/Body"

                    if target_box != actual_box:
                        serve_result = "WIDE" 
                    elif bounce_flat_y > full_court_length:
                        serve_result = "LONG"
                    elif bounce_flat_x < 0 or bounce_flat_x > full_court_width:
                        serve_result = "WIDE"
                    else:
                        serve_result = "IN"

                    print("\n==========================")
                    print("🎯 LIVE SERVE DETECTED:")
                    print(f"Target Box: {target_box} | Result: {serve_result} ({serve_placement})")
                    print("==========================\n")

                    display_bx_pixel = bx_pixel
                    display_by_pixel = by_pixel
                    display_target_box = target_box
                    display_serve_result = serve_result
                    display_serve_placement = serve_placement
                    show_display_overlay = True
                    display_countdown = 90 
                    try:
                        if firebase_admin._apps:
                            ref = db.reference('live_session/serves')
                            ref.push({
                                'timestamp': datetime.datetime.now().isoformat(),
                                'target': target_box,
                                'status': 'IN' if serve_result == 'IN' else 'OUT',
                                'result': serve_result,
                                'placement': serve_placement 
                            })
                            print("✅ Live stats updated successfully in Firebase!")
                    except Exception as e:
                        print(f"❌ Could not reach database: {e}")

      
        if len(history_raw_pixels) > 300:
            history_raw_pixels = history_raw_pixels[-15:]
            history_flat_coordinates = history_flat_coordinates[-15:]
            history_matrices = history_matrices[-15:]

 
    cv2.polylines(frame, [np.int32(current_corners)], True, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    for pt in history_raw_pixels[-15:]:
        if pt is not None:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 255), -1)
            
    if show_display_overlay:
        color = (0, 255, 0) if display_serve_result == "IN" else (0, 0, 255)
        cv2.circle(frame, (display_bx_pixel, display_by_pixel), 10, color, -1)
        
        cv2.putText(frame, f"T:{display_target_box}", (display_bx_pixel + 15, display_by_pixel - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"{display_serve_result}", (display_bx_pixel + 15, display_by_pixel - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"({display_serve_placement})", (display_bx_pixel + 15, display_by_pixel + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        display_countdown -= 1
        if display_countdown <= 0:
            show_display_overlay = False

    cv2.imshow("Live Tennis Tracker (PC Engine)", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
