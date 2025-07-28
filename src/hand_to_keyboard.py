from turtle import delay
import mediapipe as mp
import cv2
import numpy as np
import math
import time
from pynput.keyboard import Controller, Key

keyboardController = Controller()
# initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# initialize camera
cap = cv2.VideoCapture(0)

# check if camera is working
if not cap.isOpened():
    print("error: cannot open camera")
    exit()

print("hand recognition - hand posture detection with angle normalization")
print("press 'q' to quit")

# define colors for each hand
COLORS = {
    'Left': (255, 0, 0),    # blue color for left hand
    'Right': (0, 0, 255),   # red color for right hand
}


# define landmarks for each finger
FINGER_TIPS = [4, 8, 12, 16, 20]  # fingertips
FINGER_PIPS = [3, 6, 10, 14, 18]  # finger joints near tips

def calculate_hand_angle(landmarks):
    """
    calculate hand tilt angle based on vector from wrist to middle finger
    """
    wrist = landmarks[0]          # wrist
    middle_mcp = landmarks[9]     # middle finger base joint
    
    # calculate vector from wrist to middle finger
    dx = middle_mcp[0] - wrist[0]
    dy = middle_mcp[1] - wrist[1]
    
    # calculate angle (radian) relative to y-axis (upward)
    angle = math.atan2(dx, -dy)  # -dy because y coordinate increases downward
    
    return angle

def rotate_point(point, center, angle):
    """
    rotate a point around center with angle (radian)
    """
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    # convert to relative coordinates with center
    x = point[0] - center[0]
    y = point[1] - center[1]
    
    # rotate
    new_x = x * cos_angle - y * sin_angle
    new_y = x * sin_angle + y * cos_angle
    
    # convert back to original coordinates
    return [new_x + center[0], new_y + center[1]]

def normalize_hand_orientation(landmarks):
    """
    normalize hand angle to vertical direction
    """
    # calculate current hand angle
    current_angle = calculate_hand_angle(landmarks)
    
    # angle needed to rotate to vertical direction
    rotation_angle = -current_angle
    
    # use wrist as rotation center
    center = landmarks[0]
    
    # rotate all landmarks
    normalized_landmarks = []
    for landmark in landmarks:
        rotated = rotate_point(landmark, center, rotation_angle)
        normalized_landmarks.append(rotated)
    
    return normalized_landmarks

def point_in_triangle(p, a, b, c):
    """
    check if point p is inside triangle abc
    using barycentric coordinates method
    """
    x, y = p
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    
    # calculate area of triangle abc
    area_abc = abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
    
    # if area = 0 then points are collinear
    if area_abc < 1e-10:
        return False
    
    # calculate areas of sub-triangles
    area_pbc = abs((x * (y2 - y3) + x2 * (y3 - y) + x3 * (y - y2)) / 2.0)
    area_apc = abs((x1 * (y - y3) + x * (y3 - y1) + x3 * (y1 - y)) / 2.0)
    area_abp = abs((x1 * (y2 - y) + x2 * (y - y1) + x * (y1 - y2)) / 2.0)
    
    # check condition
    return abs(area_abc - (area_pbc + area_apc + area_abp)) < 1e-10

def ray_line_intersection(ray_start, ray_through, line_p1, line_p2):
    """
    calculate intersection of ray and line
    """
    x1, y1 = ray_start
    x2, y2 = ray_through
    x3, y3 = line_p1
    x4, y4 = line_p2
    
    ray_dx = x2 - x1
    ray_dy = y2 - y1
    line_dx = x4 - x3
    line_dy = y4 - y3
    
    denom = ray_dx * line_dy - ray_dy * line_dx
    
    if abs(denom) < 1e-10:
        return False, None
    
    t = ((x3 - x1) * line_dy - (y3 - y1) * line_dx) / denom
    u = ((x3 - x1) * ray_dy - (y3 - y1) * ray_dx) / denom
    
    if t >= 1.0 and 0 <= u <= 1:
        intersection_x = x1 + t * ray_dx
        intersection_y = y1 + t * ray_dy
        return True, (intersection_x, intersection_y)
    
    return False, None

def is_thumb_folded(landmarks):
    """
    check if thumb is folded using two methods:
    1. ray method: ray from 3 through 4 intersects line 0-5
    2. triangle method: landmark 3 or 4 is inside triangle 0-5-17
    """
    wrist = landmarks[0]          # wrist
    thumb_ip = landmarks[3]       # thumb joint near tip
    thumb_tip = landmarks[4]      # thumb tip
    index_mcp = landmarks[7]      # index finger base joint
    pinky_mcp = landmarks[19]     # pinky finger base joint
    
    # method 1: check ray intersection
    has_intersection, intersection = ray_line_intersection(
        thumb_ip, thumb_tip,
        wrist, index_mcp
    )
    
    # method 2: check triangle
    # if landmark 3 or 4 is inside triangle 0-5-17 then thumb is folded
    thumb_ip_in_triangle = point_in_triangle(thumb_ip, wrist, index_mcp, pinky_mcp)
    thumb_tip_in_triangle = point_in_triangle(thumb_tip, wrist, index_mcp, pinky_mcp)
    
    # thumb is considered folded if any condition is true
    return has_intersection or thumb_ip_in_triangle or thumb_tip_in_triangle

def detect_finger_posture(landmarks):
    """
    detect fingers that are extended with angle-normalized landmarks
    returns list of finger indices that are up (1-5: thumb, index, middle, ring, pinky)
    """
    # normalize hand angle before analysis
    normalized_landmarks = normalize_hand_orientation(landmarks)
    
    fingers_up = [0, 0, 0, 0, 0]
    
    # thumb - use ray + triangle logic on normalized landmarks
    if not is_thumb_folded(normalized_landmarks):
        fingers_up[0] = 1
    else:
        fingers_up[0] = 0
    
    # other 4 fingers - compare on normalized landmarks
    for i in range(1, 5):
        tip_y = normalized_landmarks[FINGER_TIPS[i]][1]
        pip_y = normalized_landmarks[FINGER_PIPS[i]][1]
        
        # after normalization, extended finger will have tip_y < pip_y
        if tip_y < pip_y:
            fingers_up[i] = 1
        else:
            fingers_up[i] = 0
    
    return fingers_up
    

KEY_MAP = {
    0: {        #left hand
        0: "q",     #thumb
        1: "w",     #index
        2: "a",     #middle
        3: "s",     #ring
        4: "d"      #pinky
    },
    1: {        #right hand
        0: "e",     #thumb
        1: "1",     #index
        2: "2",     #middle
        3: "3",     #ring
        4: "4"      #pinky
    }
}

def get_mapped_key(hands_array):
    """
    return: (left_key, right_key, special_key) 
    mapped by fingers
    """
    left_key = ""
    right_key = ""
    special_key = ""
    
    # Check left hand gestures
    if np.array_equal(hands_array[0], [1, 0, 0, 0, 0]):
        left_key = KEY_MAP[0][0]
    elif np.array_equal(hands_array[0], [0, 1, 0, 0, 0]):
        left_key = KEY_MAP[0][1]
    elif np.array_equal(hands_array[0], [0, 1, 1, 0, 0]):
        left_key = KEY_MAP[0][2]
    elif np.array_equal(hands_array[0], [0, 1, 1, 1, 0]):
        left_key = KEY_MAP[0][3]
    elif np.array_equal(hands_array[0], [0, 1, 1, 1, 1]):
        left_key = KEY_MAP[0][4]
    
    # Check right hand gestures
    if np.array_equal(hands_array[1], [1, 0, 0, 0, 0]):
        right_key = KEY_MAP[1][0]
    elif np.array_equal(hands_array[1], [0, 1, 0, 0, 0]):
        right_key = KEY_MAP[1][1]
    elif np.array_equal(hands_array[1], [0, 1, 1, 0, 0]):
        right_key = KEY_MAP[1][2]
    elif np.array_equal(hands_array[1], [0, 1, 1, 1, 0]):
        right_key = KEY_MAP[1][3]
    elif np.array_equal(hands_array[1], [0, 1, 1, 1, 1]):
        right_key = KEY_MAP[1][4]
    
    # Check special gestures
    if np.all(hands_array == 1):
        special_key = "SPACE"
    elif np.array_equal(hands_array[0], [0, 1, 0, 0, 1]) or np.array_equal(hands_array[1], [0, 1, 0, 0, 1]):
        special_key = "ESC"
    
    return left_key, right_key, special_key

def hand_to_keyboard(hands_array):
    if np.array_equal(hands_array[0], [1, 0, 0, 0, 0]):
        keyboardController.type(KEY_MAP[0][0])
    elif np.array_equal(hands_array[0], [0, 1, 0, 0, 0]):
        keyboardController.type(KEY_MAP[0][1])
    elif np.array_equal(hands_array[0], [0, 1, 1, 0, 0]):
        keyboardController.type(KEY_MAP[0][2])
    elif np.array_equal(hands_array[0], [0, 1, 1, 1, 0]):
        keyboardController.type(KEY_MAP[0][3])
    elif np.array_equal(hands_array[0], [0, 1, 1, 1, 1]):
        keyboardController.type(KEY_MAP[0][4])
    
    if np.array_equal(hands_array[1], [1, 0, 0, 0, 0]):
        keyboardController.type(KEY_MAP[1][0])
    elif np.array_equal(hands_array[1], [0, 1, 0, 0, 0]):
        keyboardController.type(KEY_MAP[1][1])
    elif np.array_equal(hands_array[1], [0, 1, 1, 0, 0]):
        keyboardController.type(KEY_MAP[1][2])
    elif np.array_equal(hands_array[1], [0, 1, 1, 1, 0]):
        keyboardController.type(KEY_MAP[1][3])
    elif np.array_equal(hands_array[1], [0, 1, 1, 1, 1]):
        keyboardController.type(KEY_MAP[1][4])
    if np.all(hands_array == 1):
        keyboardController.press(Key.space)

    if np.array_equal(hands_array[0], [0, 1, 0, 0, 1]) or np.array_equal(hands_array[1], [0, 1, 0, 0, 1]):
        keyboardController.press(Key.esc)

counter = 0
while True:
    counter += 1
    ret, frame = cap.read()
    
    if not ret:
        print("error: cannot read frame from camera")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    hand_count = {'Left': 0, 'Right': 0}
    
    # 2x5 array for both hands (Left=row0, Right=row1)
    hands_array = np.zeros((2, 5), dtype=int)
    
    # draw landmarks if hands detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            # get left or right hand information
            hand_label = handedness.classification[0].label
            hand_score = handedness.classification[0].score
            
            # count number of hands
            hand_count[hand_label] += 1
            color = COLORS[hand_label]
            
            # draw landmarks and connections with corresponding color
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=color, thickness=2)
            )
            
            # get landmark coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                height, width, _ = frame.shape
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks.append([x, y])
            
            # calculate hand angle for disp
            hand_angle = calculate_hand_angle(landmarks)
            angle_degrees = math.degrees(hand_angle)
            
            # detect finger posture
            fingers_up = detect_finger_posture(landmarks)
            
            # Update hands_array based on hand_label
            row_idx = 0 if hand_label == 'Left' else 1  # Left=0, Right=1
            hands_array[row_idx] = fingers_up
            # draw bounding box around hand
            if landmarks:
                x_coords = [point[0] for point in landmarks]
                y_coords = [point[1] for point in landmarks]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # add padding to bounding box
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(width, x_max + padding)
                y_max = min(height, y_max + padding)
                
                # draw bounding box with corresponding color
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
                
                # disp hand label with posture
                main_label = f"{hand_label} ({hand_score:.2f})"
                posture_label = f"{hand_label}{fingers_up}"
                angle_label = f"angle: {angle_degrees:.1f}deg"
                
                # draw bg for text
                text_size = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x_min, y_min - 85), (x_min + max(text_size[0], 200) + 10, y_min), color, -1)
                
                # disp text with white color on colored bg
                cv2.putText(frame, main_label, (x_min + 5, y_min - 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, posture_label, (x_min + 5, y_min - 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, angle_label, (x_min + 5, y_min - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # disp number if multiple hands of same type
                if hand_count[hand_label] > 1:
                    number_text = f"#{hand_count[hand_label]}"
                    cv2.putText(frame, number_text, (x_max - 30, y_min + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # draw finger numbers on landmarks (removed green circles)
                for finger_idx in fingers_up:
                    tip_idx = FINGER_TIPS[finger_idx - 1]
                    tip_pos = landmarks[tip_idx]
                    cv2.putText(frame, str(finger_idx), (tip_pos[0] - 5, tip_pos[1] + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # draw hand direction line (from wrist to middle finger)
                wrist_pos = tuple(landmarks[0])
                middle_mcp_pos = tuple(landmarks[9])
                cv2.line(frame, wrist_pos, middle_mcp_pos, (255, 255, 255), 2)
                cv2.arrowedLine(frame, wrist_pos, middle_mcp_pos, (255, 255, 255), 2, tipLength=0.1)
    
    # Get mapped keys for disp
    left_key, right_key, special_key = get_mapped_key(hands_array)
    
    if counter >= 10 and hands_array.any():
        # debugging
        # print(hands_array)
        # print(counter)
        counter = 0
        hand_to_keyboard(hands_array)
    
    # disp hand stats
    total_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    cv2.putText(frame, f"total hands: {total_hands}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # disp count for each hand type
    cv2.putText(frame, f"left hands: {hand_count['Left']}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['Left'], 2)
    cv2.putText(frame, f"right hands: {hand_count['Right']}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['Right'], 2)
    
    # disp instructions
    cv2.putText(frame, "press 'q' to quit", (10, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # disp mapped keys
    if left_key:
        cv2.putText(frame, f"Left Hand -> {left_key.upper()}", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['Left'], 2)
    if right_key:
        cv2.putText(frame, f"Right Hand -> {right_key.upper()}", (10, 190), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['Right'], 2)
    if special_key:
        cv2.putText(frame, f"Special -> {special_key}", (10, 220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # explaination
    cv2.putText(frame, "[1=thumb, 2=index, 3=middle, 4=ring, 5=pinky] - angle normalized", (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('hand recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("hand recognition stopped")