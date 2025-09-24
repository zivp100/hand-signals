import streamlit as st
import time
import cv2
import numpy as np
from datetime import datetime
import os
import mediapipe as mp
from PIL import Image

def main():
    # Initialize session state first
    if 'countdown_active' not in st.session_state:
        st.session_state.countdown_active = False
    if 'countdown_value' not in st.session_state:
        st.session_state.countdown_value = 3
    if 'photo_taken' not in st.session_state:
        st.session_state.photo_taken = False
    if 'captured_image' not in st.session_state:
        st.session_state.captured_image = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None
    if 'angle_image' not in st.session_state:
        st.session_state.angle_image = None
    if 'finger_angle' not in st.session_state:
        st.session_state.finger_angle = None
    
    st.title("👉 Which direction am I pointing")
    st.markdown("**Use your pointing finger to show the desired direction**")
    
    
    # Create columns for layout
    # Camera and countdown layout
    if not st.session_state.photo_taken:
        # Create two columns: camera on left, countdown on right
        cam_col, timer_col = st.columns([2, 1])
        
        with cam_col:
            st.markdown("### 📷 Camera Preview")
            
            # Show instructions when countdown is active
            if st.session_state.countdown_active:
                st.info("🎯 **Get Ready!** Position your hand in the camera view. Photo will be taken automatically when countdown reaches zero!")
            
            camera_photo = st.camera_input("Live camera preview (photo will be taken automatically)")
        
        with timer_col:
            st.markdown("### ⏰ Timer")
            # Display countdown timer
            if st.session_state.countdown_active:
                st.markdown(f"# {st.session_state.countdown_value}")
                st.markdown("### Get ready! 📸")
            else:
                st.markdown("# 3")
                st.markdown("### Click 'Start Timer' to begin")
                # Start Timer button
                if st.button("⏰ Start Timer", type="primary", use_container_width=True):
                    st.session_state.countdown_active = True
                    st.rerun()
                
                # Take Photo button
                if st.button("📸 Take Photo", use_container_width=True):
                    # Use OpenCV to capture current camera image
                    try:
                        # Capture image using OpenCV
                        cap = cv2.VideoCapture(0)
                        if cap.isOpened():
                            # Warm up camera
                            for _ in range(5):
                                ret, frame = cap.read()
                                if ret:
                                    time.sleep(0.1)
                            
                            # Capture final frame
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret and frame is not None:
                                # Convert BGR to RGB
                                image_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Store the captured image
                                st.session_state.captured_image = image_array
                                st.session_state.photo_taken = True
                                
                                # Process hand landmarks
                                detect_hand_landmarks(image_array)
                                
                                # Process index finger line
                                if st.session_state.annotated_image is not None:
                                    # Get landmarks from the last detection
                                    mp_hands = mp.solutions.hands
                                    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                                        results = hands.process(cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                                        if results.multi_hand_landmarks:
                                            landmarks = results.multi_hand_landmarks[0]
                                            angle_image, finger_angle = draw_index_finger_line(image_array, landmarks)
                                            st.session_state.angle_image = angle_image
                                            st.session_state.finger_angle = finger_angle
                                
                                st.rerun()
                            else:
                                st.error("❌ Failed to capture image from camera")
                        else:
                            st.error("❌ Could not access camera")
                    except Exception as e:
                        st.error(f"❌ Error capturing image: {str(e)}")
        
        # Process manual photo if taken (fallback option)
        if camera_photo is not None:
            # Convert the photo to the format we need
            image = Image.open(camera_photo)
            image_array = np.array(image)
            
            # Store the captured image
            st.session_state.captured_image = image_array
            st.session_state.photo_taken = True
            
            # Detect hand landmarks
            st.info("🔍 Detecting hand landmarks...")
            annotated_image = detect_hand_landmarks(image_array)
            st.session_state.annotated_image = annotated_image
            
            # Save images to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            annotated_filename = f"photo_annotated_{timestamp}.jpg"
            filepath = os.path.join("photos", filename)
            annotated_filepath = os.path.join("photos", annotated_filename)
            
            # Create photos directory if it doesn't exist
            os.makedirs("photos", exist_ok=True)
            
            # Save original image
            image.save(filepath)
            
            # Save annotated image if available
            if annotated_image is not None:
                annotated_pil = Image.fromarray(annotated_image)
                annotated_pil.save(annotated_filepath)
                st.success(f"✅ Photos saved as {filename} and {annotated_filename}")
            else:
                st.success(f"✅ Photo saved as {filename}")
                st.warning("⚠️ No hand landmarks detected in the image")
            
            # Stop countdown if it was active
            if st.session_state.countdown_active:
                st.session_state.countdown_active = False
    else:
        # Photo taken - show success message
        st.markdown("## ✅ Photo Taken!")
        st.markdown("### Check the results below")
    
    # Stop countdown button (only when countdown is active)
    if st.session_state.countdown_active:
        if st.button("⏹️ Stop Countdown", use_container_width=True):
            st.session_state.countdown_active = False
            st.session_state.countdown_value = 3
            st.rerun()
    
    # Countdown logic
    if st.session_state.countdown_active:
        if st.session_state.countdown_value > 0:
            time.sleep(1)
            st.session_state.countdown_value -= 1
            st.rerun()
        else:
            # Timer reached zero - automatically take photo
            st.session_state.countdown_active = False
            st.session_state.photo_taken = True
            take_photo_automatically()
            st.rerun()
    
    # Display captured image
    if st.session_state.photo_taken and st.session_state.captured_image is not None:
        st.markdown("---")
        st.markdown("## 📸 Your Photo!")
        
        # Create four columns for original, annotated, angle images, and direction arrow
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("#### Original")
        with col2:
            st.markdown("#### Hand Landmarks")    
        with col3:
            st.markdown("#### Index Finger")
        with col4:
            st.markdown("#### Direction Arrow")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image(st.session_state.captured_image, caption=f"Captured at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", use_container_width=True)
            # Add empty space to match other columns
            st.markdown("&nbsp;")
        
        with col2:
            if st.session_state.annotated_image is not None:
                st.image(st.session_state.annotated_image, caption="Hand landmarks detected", use_container_width=True)
            else:
                # Placeholder to maintain alignment
                st.image(st.session_state.captured_image, caption="Processing hand landmarks...", use_container_width=True)
            # Add empty space to match other columns
            st.markdown("&nbsp;")
        
        with col3:
            if st.session_state.angle_image is not None:
                st.image(st.session_state.angle_image, caption="Index finger line (landmarks 5-8)", use_container_width=True)
                #if st.session_state.finger_angle is not None:
                #    st.markdown(f"**Angle: {st.session_state.finger_angle:.1f}°**")
                #else:
                #    st.info("No finger detected")
            else:
                # Placeholder to maintain alignment
                st.image(st.session_state.captured_image, caption="No hand detected", use_container_width=True)
                # Add empty space to match other columns
                st.markdown("&nbsp;")
        
        with col4:
            if st.session_state.finger_angle is not None:
                # Create arrow image
                arrow_image = create_direction_arrow(st.session_state.finger_angle)
                st.image(arrow_image, caption="Pointing direction", use_container_width=True)
            else:
                # Placeholder to maintain alignment
                st.image(st.session_state.captured_image, caption="No hand detected", use_container_width=True)
                # Add empty space to match other columns
                st.markdown("&nbsp;")
        
        # Reset button
        if st.button("🔄 Take Another Photo", type="primary"):
            # Clear all session state variables
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            # Force a complete rerun
            st.rerun()


def detect_hand_landmarks(image):
    """Detect hand landmarks using MediaPipe and draw them on the image"""
    try:
        # Initialize MediaPipe hands
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        # Convert RGB to BGR for MediaPipe
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        ) as hands:
            # Process the image
            results = hands.process(image_bgr)
            
            # Create a copy for drawing
            annotated_image = image_bgr.copy()
            
            # Draw hand landmarks and create index finger line
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Create index finger line image and calculate angle
                    line_image, angle = draw_index_finger_line(image_bgr, hand_landmarks.landmark)
                    st.session_state.angle_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
                    st.session_state.finger_angle = angle
            
            # Convert back to RGB for Streamlit
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            return annotated_image_rgb
            
    except Exception as e:
        st.error(f"❌ Error detecting hand landmarks: {str(e)}")
        return None

def draw_index_finger_line(image, landmarks):
    """Draw a line connecting landmarks 5, 6, 7, 8 (index finger) and calculate angle"""
    try:
        # Create a copy of the image
        line_image = image.copy()
        
        if landmarks is None or len(landmarks) < 9:
            return line_image, None
        
        # Get image dimensions
        height, width = line_image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for i in [5, 6, 7, 8]:  # Index finger landmarks
            if i < len(landmarks):
                x = int(landmarks[i].x * width)
                y = int(landmarks[i].y * height)
                points.append((x, y))
        
        if len(points) < 4:
            return line_image, None
        
        # Draw the line connecting all four points
        for i in range(len(points) - 1):
            cv2.line(line_image, points[i], points[i + 1], (0, 255, 0), 8)
        
        # Draw circles at each landmark
        for point in points:
            cv2.circle(line_image, point, 5, (255, 0, 0), -1)
        
        # Calculate angle of the line from first to last point
        start_point = points[0]  # Landmark 5
        end_point = points[-1]   # Landmark 8
        
        # Calculate angle in degrees
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalize angle to 0-360 degrees
        if angle < 0:
            angle += 360
        
        # Add angle text to image
        #cv2.putText(line_image, f"Angle: {angle:.1f}°", (10, 30), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return line_image, angle
        
    except Exception as e:
        st.error(f"❌ Error drawing finger line: {str(e)}")
        return image, None

def create_direction_arrow(angle):
    """Create an arrow image pointing in the calculated direction"""
    try:
        # Create a white background image
        size = 300
        arrow_image = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        # Calculate arrow direction (convert angle to radians)
        angle_rad = np.radians(angle)
        
        # Center of the image
        center_x, center_y = size // 2, size // 2
        
        # Arrow length
        arrow_length = 100
        
        # Calculate arrow end point
        end_x = int(center_x + arrow_length * np.cos(angle_rad))
        end_y = int(center_y + arrow_length * np.sin(angle_rad))
        
        # Draw the main arrow line
        cv2.arrowedLine(arrow_image, (center_x, center_y), (end_x, end_y), (0, 0, 0), 8, tipLength=0.3)
        
        # Add angle text
        cv2.putText(arrow_image, f"{angle:.1f}°", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return arrow_image
        
    except Exception as e:
        st.error(f"❌ Error creating direction arrow: {str(e)}")
        # Return a simple white image as fallback
        return np.ones((300, 300, 3), dtype=np.uint8) * 255

def take_photo_automatically():
    """Automatically capture photo using OpenCV when countdown reaches zero"""
    try:
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Could not access camera. Please make sure your camera is connected and not being used by another application.")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Allow camera to warm up by reading a few frames
        for _ in range(5):
            ret, _ = cap.read()
            if not ret:
                break
            time.sleep(0.1)  # Small delay between frames
        
        # Capture the actual frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None and frame.size > 0:
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.captured_image = frame_rgb
            
            # Detect hand landmarks
            st.info("🔍 Detecting hand landmarks...")
            annotated_image = detect_hand_landmarks(frame_rgb)
            st.session_state.annotated_image = annotated_image
            
            # Save images to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            annotated_filename = f"photo_annotated_{timestamp}.jpg"
            filepath = os.path.join("photos", filename)
            annotated_filepath = os.path.join("photos", annotated_filename)
            
            # Create photos directory if it doesn't exist
            os.makedirs("photos", exist_ok=True)
            
            # Save original image
            cv2.imwrite(filepath, frame)
            
            # Save annotated image if available
            if annotated_image is not None:
                annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(annotated_filepath, annotated_bgr)
                
                # Save angle image if available
                if st.session_state.angle_image is not None:
                    angle_filename = f"photo_angle_{timestamp}.jpg"
                    angle_filepath = os.path.join("photos", angle_filename)
                    angle_bgr = cv2.cvtColor(st.session_state.angle_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(angle_filepath, angle_bgr)
                    st.success(f"✅ Photos saved as {filename}, {annotated_filename}, and {angle_filename}")
                else:
                    st.success(f"✅ Photos saved as {filename} and {annotated_filename}")
            else:
                st.success(f"✅ Photo saved as {filename}")
                st.warning("⚠️ No hand landmarks detected in the image")
        else:
            st.error("❌ Failed to capture image - camera may not be ready")
            
    except Exception as e:
        st.error(f"❌ Error capturing image: {str(e)}")

def take_picture():
    """This function is now handled by Streamlit's camera input"""
    pass

if __name__ == "__main__":
    main()
