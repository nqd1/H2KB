# H2KB - Hand to Keyboard

H2KB is a real-time hand gesture recognition application that maps finger postures to keyboard key presses.

- **Hand Landmark Detection:** Uses **MediaPipe** to detect and extract 21 hand landmarks from the webcam feed.
- **Angle Normalization:** Calculates the tilt angle of the hand and normalizes its orientation, ensuring gesture recognition is robust regardless of hand rotation.
- **Finger State Analysis:** Determines which fingers are extended by comparing the positions of specific landmarks for each finger.
- **Gesture to Key Mapping:** Represents the state of the fingers as an array (e.g., `[0, 1, 1, 0, 0]`) and maps each unique posture to a corresponding keyboard key.
- **Keyboard Event Emission:** Utilizes **pynput** to send keyboard events to the operating system, enabling gesture-based control of other applications.
- **Visual Feedback:** Displays a live video feed with hand bounding boxes, left/right hand labels, finger states, hand angle, and the currently triggered key for intuitive user feedback.
- **Fast and Simple:** Designed for speed and simplicity, supporting both hands simultaneously for versatile control.

**Requirements:** Python, OpenCV, MediaPipe, numpy, pynput.

**Applications:** Touchless computer control, gaming, accessibility for people with disabilities, AI demonstrations, and more.

**Install:**
```
git clone https://github.com/yourusername/H2KB.git
cd H2KB
```