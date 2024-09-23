# import cv2
# import mediapipe as mp
# import pyautogui
# import time
#
#
# def count_fingers(lst):
#     cnt = 0
#
#     thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2
#
#     if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
#         cnt += 1
#
#     if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
#         cnt += 1
#
#     if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
#         cnt += 1
#
#     if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
#         cnt += 1
#
#     if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
#         cnt += 1
#
#     return cnt
#
#
# cap = cv2.VideoCapture(0)
#
# drawing = mp.solutions.drawing_utils
# hands = mp.solutions.hands
# hand_obj = hands.Hands(max_num_hands=1)
#
# start_init = False
#
# prev = -1
#
# while True:
#     end_time = time.time()
#     _, frm = cap.read()
#     frm = cv2.flip(frm, 1)
#
#     res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
#
#     if res.multi_hand_landmarks:
#
#         hand_keyPoints = res.multi_hand_landmarks[0]
#
#         cnt = count_fingers(hand_keyPoints)
#
#         if not (prev == cnt):
#             if not (start_init):
#                 start_time = time.time()
#                 start_init = True
#
#             elif (end_time - start_time) > 0.2:
#                 if (cnt == 1):
#                     pyautogui.press("right")
#
#                 elif (cnt == 2):
#                     pyautogui.press("left")
#
#                 elif (cnt == 3):
#                     pyautogui.press("up")
#
#                 elif (cnt == 4):
#                     pyautogui.press("down")
#
#                 elif (cnt == 5):
#                     pyautogui.press("space")
#
#                 prev = cnt
#                 start_init = False
#
#         drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)
#
#     cv2.imshow("window", frm)
#
#     if cv2.waitKey(1) == 27:
#         cv2.destroyAllWindows()
#         cap.release()
#         break


# With GUI ->

# import cv2
# import mediapipe as mp
# import pyautogui
# import time
# import tkinter as tk
# from tkinter import ttk
#
# def count_fingers(lst):
#     cnt = 0
#     thresh = (lst.landmark[0].y * 100 - lst.landmark[9].y * 100) / 2
#     if (lst.landmark[5].y * 100 - lst.landmark[8].y * 100) > thresh:
#         cnt += 1
#     if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
#         cnt += 1
#     if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
#         cnt += 1
#     if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
#         cnt += 1
#     if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) > 6:
#         cnt += 1
#     return cnt
#
# class GestureControlGUI:
#     def __init__(self, master):
#         self.master = master
#         master.title("Gesture Control GUI")
#         master.geometry("400x300")
#
#         self.gesture_actions = {
#             1: "right",
#             2: "left",
#             3: "up",
#             4: "down",
#             5: "space"
#         }
#
#         self.create_widgets()
#
#     def create_widgets(self):
#         # Create and place labels and comboboxes for each gesture
#         for i in range(1, 6):
#             label = ttk.Label(self.master, text=f"{i} Finger{'s' if i > 1 else ''}:")
#             label.grid(row=i-1, column=0, padx=5, pady=5, sticky="e")
#
#             combobox = ttk.Combobox(self.master, values=list(pyautogui.KEYBOARD_KEYS))
#             combobox.set(self.gesture_actions[i])
#             combobox.grid(row=i-1, column=1, padx=5, pady=5)
#             combobox.bind("<<ComboboxSelected>>", lambda event, i=i: self.update_action(event, i))
#
#         # Start button
#         self.start_button = ttk.Button(self.master, text="Start Gesture Control", command=self.start_gesture_control)
#         self.start_button.grid(row=5, column=0, columnspan=2, pady=20)
#
#     def update_action(self, event, fingers):
#         self.gesture_actions[fingers] = event.widget.get()
#
#     def start_gesture_control(self):
#         self.master.iconify()  # Minimize the GUI window
#         self.run_gesture_control()
#
#     def run_gesture_control(self):
#         cap = cv2.VideoCapture(0)
#         drawing = mp.solutions.drawing_utils
#         hands = mp.solutions.hands
#         hand_obj = hands.Hands(max_num_hands=1)
#
#         start_init = False
#         prev = -1
#
#         while True:
#             end_time = time.time()
#             _, frm = cap.read()
#             frm = cv2.flip(frm, 1)
#
#             res = hand_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
#
#             if res.multi_hand_landmarks:
#                 hand_keyPoints = res.multi_hand_landmarks[0]
#                 cnt = count_fingers(hand_keyPoints)
#
#                 if not (prev == cnt):
#                     if not (start_init):
#                         start_time = time.time()
#                         start_init = True
#                     elif (end_time - start_time) > 0.2:
#                         if cnt in self.gesture_actions:
#                             pyautogui.press(self.gesture_actions[cnt])
#                         prev = cnt
#                         start_init = False
#
#                 drawing.draw_landmarks(frm, hand_keyPoints, hands.HAND_CONNECTIONS)
#
#             cv2.imshow("Gesture Control", frm)
#
#             if cv2.waitKey(1) == 27:  # ESC key
#                 break
#
#         cv2.destroyAllWindows()
#         cap.release()
#         self.master.deiconify()  # Restore the GUI window
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = GestureControlGUI(root)
#     root.mainloop()

# Gesture Control Improved using Angle btw Joints(REJECTED)->

# import cv2
# import mediapipe as mp
# import pyautogui
# import time
# import tkinter as tk
# from tkinter import ttk
# import numpy as np
#
#
# class GestureControlGUI:
#     def __init__(self, master):
#         self.master = master
#         master.title("Gesture Control GUI")
#         master.geometry("500x400")
#
#         self.gesture_actions = {
#             1: "right",
#             2: "left",
#             3: "up",
#             4: "down",
#             5: "space"
#         }
#
#         self.create_widgets()
#
#     def create_widgets(self):
#         # Create and place labels and comboboxes for each gesture
#         for i in range(1, 6):
#             label = ttk.Label(self.master, text=f"{i} Finger{'s' if i > 1 else ''}:")
#             label.grid(row=i - 1, column=0, padx=5, pady=5, sticky="e")
#
#             combobox = ttk.Combobox(self.master, values=list(pyautogui.KEYBOARD_KEYS))
#             combobox.set(self.gesture_actions[i])
#             combobox.grid(row=i - 1, column=1, padx=5, pady=5)
#             combobox.bind("<<ComboboxSelected>>", lambda event, i=i: self.update_action(event, i))
#
#         # Sensitivity slider
#         self.sensitivity_label = ttk.Label(self.master, text="Gesture Sensitivity:")
#         self.sensitivity_label.grid(row=5, column=0, padx=5, pady=5, sticky="e")
#         self.sensitivity_slider = ttk.Scale(self.master, from_=0.1, to=1.0, orient="horizontal", length=200)
#         self.sensitivity_slider.set(0.5)
#         self.sensitivity_slider.grid(row=5, column=1, padx=5, pady=5)
#
#         # Smoothing factor slider
#         self.smoothing_label = ttk.Label(self.master, text="Smoothing Factor:")
#         self.smoothing_label.grid(row=6, column=0, padx=5, pady=5, sticky="e")
#         self.smoothing_slider = ttk.Scale(self.master, from_=1, to=10, orient="horizontal", length=200)
#         self.smoothing_slider.set(3)
#         self.smoothing_slider.grid(row=6, column=1, padx=5, pady=5)
#
#         # Start button
#         self.start_button = ttk.Button(self.master, text="Start Gesture Control", command=self.start_gesture_control)
#         self.start_button.grid(row=7, column=0, columnspan=2, pady=20)
#
#     def update_action(self, event, fingers):
#         self.gesture_actions[fingers] = event.widget.get()
#
#     def start_gesture_control(self):
#         self.master.iconify()  # Minimize the GUI window
#         self.run_gesture_control()
#
#     def get_finger_angles(self, hand_landmarks):
#         angles = []
#         for i in range(5):
#             base = hand_landmarks.landmark[0]
#             mcp = hand_landmarks.landmark[i * 4 + 1]
#             tip = hand_landmarks.landmark[i * 4 + 4]
#
#             v1 = np.array([mcp.x - base.x, mcp.y - base.y, mcp.z - base.z])
#             v2 = np.array([tip.x - mcp.x, tip.y - mcp.y, tip.z - mcp.z])
#
#             angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
#             angles.append(angle)
#         return angles
#
#     def count_fingers(self, angles, threshold):
#         return sum(1 for angle in angles if angle > threshold)
#
#     def run_gesture_control(self):
#         cap = cv2.VideoCapture(0)
#         mp_hands = mp.solutions.hands
#         hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
#                                min_tracking_confidence=0.7)
#         mp_draw = mp.solutions.drawing_utils
#
#         prev_cnt = 0
#         gesture_buffer = []
#         buffer_size = 5
#         last_action_time = time.time()
#         action_cooldown = 0.5  # seconds
#
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             frame = cv2.flip(frame, 1)
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb_frame)
#
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#                     angles = self.get_finger_angles(hand_landmarks)
#                     threshold = self.sensitivity_slider.get() * 90  # Convert sensitivity to angle threshold
#                     cnt = self.count_fingers(angles, threshold)
#
#                     gesture_buffer.append(cnt)
#                     if len(gesture_buffer) > buffer_size:
#                         gesture_buffer.pop(0)
#
#                     smoothing_factor = int(self.smoothing_slider.get())
#                     smoothed_cnt = max(set(gesture_buffer), key=gesture_buffer.count)
#
#                     current_time = time.time()
#                     if smoothed_cnt != prev_cnt and current_time - last_action_time > action_cooldown:
#                         if smoothed_cnt in self.gesture_actions:
#                             pyautogui.press(self.gesture_actions[smoothed_cnt])
#                             last_action_time = current_time
#                         prev_cnt = smoothed_cnt
#
#                     # Display finger count on the frame
#                     cv2.putText(frame, f"Fingers: {smoothed_cnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
#                                 2)
#
#             cv2.imshow("Gesture Control", frame)
#
#             if cv2.waitKey(1) & 0xFF == 27:  # ESC key
#                 break
#
#         cap.release()
#         cv2.destroyAllWindows()
#         self.master.deiconify()  # Restore the GUI window
#
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = GestureControlGUI(root)
#     root.mainloop()

# Gesture Control Improved - 2 ->

import cv2
import mediapipe as mp
import pyautogui
import time
import tkinter as tk
from tkinter import ttk


class GestureControlGUI:
    def __init__(self, master):
        self.master = master
        master.title("Gesture Control GUI")
        master.geometry("500x400")

        self.gesture_actions = {
            1: "right",
            2: "left",
            3: "up",
            4: "down",
            5: "space"
        }

        self.create_widgets()

    def create_widgets(self):
        # Create and place labels and comboboxes for each gesture
        for i in range(1, 6):
            label = ttk.Label(self.master, text=f"{i} Finger{'s' if i > 1 else ''}:")
            label.grid(row=i - 1, column=0, padx=5, pady=5, sticky="e")

            combobox = ttk.Combobox(self.master, values=list(pyautogui.KEYBOARD_KEYS))
            combobox.set(self.gesture_actions[i])
            combobox.grid(row=i - 1, column=1, padx=5, pady=5)
            combobox.bind("<<ComboboxSelected>>", lambda event, i=i: self.update_action(event, i))

        # Threshold slider
        self.threshold_label = ttk.Label(self.master, text="Detection Threshold:")
        self.threshold_label.grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.threshold_slider = ttk.Scale(self.master, from_=0, to=100, orient="horizontal", length=200)
        self.threshold_slider.set(50)
        self.threshold_slider.grid(row=5, column=1, padx=5, pady=5)

        # Start button
        self.start_button = ttk.Button(self.master, text="Start Gesture Control", command=self.start_gesture_control)
        self.start_button.grid(row=6, column=0, columnspan=2, pady=20)

    def update_action(self, event, fingers):
        self.gesture_actions[fingers] = event.widget.get()

    def start_gesture_control(self):
        self.master.iconify()  # Minimize the GUI window
        self.run_gesture_control()

    def count_fingers(self, hand_landmarks):
        finger_tips = [4, 8, 12, 16, 20]
        finger_anchors = [2, 6, 10, 14, 18]
        thumb_anchor = 2
        threshold = self.threshold_slider.get() / 100

        cnt = 0
        for tip, anchor in zip(finger_tips[1:], finger_anchors[1:]):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[anchor].y:
                cnt += 1

        # Special case for thumb
        if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[thumb_anchor].x:
            cnt += 1

        return cnt

    def run_gesture_control(self):
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7,
                               min_tracking_confidence=0.7)
        mp_draw = mp.solutions.drawing_utils

        prev_cnt = 0
        last_action_time = time.time()
        action_cooldown = 0.3  # seconds

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    cnt = self.count_fingers(hand_landmarks)

                    current_time = time.time()
                    if cnt != prev_cnt and current_time - last_action_time > action_cooldown:
                        if cnt in self.gesture_actions:
                            pyautogui.press(self.gesture_actions[cnt])
                            last_action_time = current_time
                        prev_cnt = cnt

                    # Display finger count on the frame
                    cv2.putText(frame, f"Fingers: {cnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()
        self.master.deiconify()  # Restore the GUI window


if __name__ == "__main__":
    root = tk.Tk()
    app = GestureControlGUI(root)
    root.mainloop()