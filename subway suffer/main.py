import cv2
import mediapipe as mp
import time
import pyautogui

# Define key mappings
keys_map = {
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down"
}

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

tipIds = [4, 8, 12, 16, 20]
video = cv2.VideoCapture(0)

# Initialize key states
current_key_pressed = set()

with mp_hand.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:
    while True:
        keyPressed = False
        ret, image = video.read()
        if not ret:
            break  # Break the loop if no frame is captured

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lmList = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                myHands = results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHands.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx, cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
                fingers = []

        if len(lmList) != 0:
            # Shifting Left (Thumb Gesture)
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                pyautogui.press(keys_map["left"])
                current_key_pressed.add(keys_map["left"])
                keyPressed = True

            # Shifting Right (Pinky Gesture)
            if lmList[tipIds[4]][1] > lmList[tipIds[4]-1][1]:
                pyautogui.press(keys_map["right"])
                current_key_pressed.add(keys_map["right"])
                keyPressed = True

            # Jumping (Open Hand Gesture)
            if all(lm[2] < lmList[i - 1][2] for i, lm in enumerate(lmList) if i > 0 and i < 5):
                pyautogui.press(keys_map["up"])
                current_key_pressed.add(keys_map["up"])
                keyPressed = True

            # Rolling Down (Closed Fist Gesture)
            if all(lm[2] > lmList[i - 1][2] for i, lm in enumerate(lmList) if i > 0 and i < 5):
                pyautogui.press(keys_map["down"])
                current_key_pressed.add(keys_map["down"])
                keyPressed = True

        # Release all keys if no gesture is detected
        if not keyPressed:
            for key in current_key_pressed:
                pyautogui.keyUp(key)
            current_key_pressed.clear()

        cv2.imshow("Frame", image)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

        # Check if the user pressed the close (X) button
        if cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE) < 1:
            break

        time.sleep(0.05)  # Add a small delay to prevent rapid key presses

# Release the video capture object
video.release()
cv2.destroyAllWindows()

