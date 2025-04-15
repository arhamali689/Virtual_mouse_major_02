import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import threading
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize hand detector
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_y = 0

# Global variable for voice command
command = ""

# Function for voice command recognition
def recognize_speech():
    global command
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        while True:  # Run in background thread
            print("Listening for commands...")
            recognizer.adjust_for_ambient_noise(source)  # Reduce noise
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command recognized: {command}")
            except sr.UnknownValueError:
                print("Could not understand the command.")
                command = ""
            except sr.RequestError:
                print("Speech Recognition service error.")
                command = ""
            time.sleep(2)  # Wait before next command to avoid overload

# Start voice recognition in background thread
thread = threading.Thread(target=recognize_speech, daemon=True)
thread.start()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index Finger
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y

                if id == 4:  # Thumb
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

                    if abs(index_y - thumb_y) < 20:
                        pyautogui.click()
                        pyautogui.sleep(1)
                    elif abs(index_y - thumb_y) < 100:
                        pyautogui.moveTo(index_x, index_y)

    # Process voice command (without blocking video stream)
    if command:
        if "click" in command:
            pyautogui.click()
        elif "right click" in command:
            pyautogui.rightClick()
        elif "scroll up" in command:
            pyautogui.scroll(500)
        elif "scroll down" in command:
            pyautogui.scroll(-500)
        elif "drag" in command:
            pyautogui.mouseDown()
        elif "drop" in command:
            pyautogui.mouseUp()
        command = ""  # Reset command after execution

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()