import cv2
import mediapipe as mp
from mediapipe.tasks import python
import threading 
from audio_controls import mute_audio, unmute_audio, set_full_volume, set_half_volume, set_qtr_volume  

class GestureRecognizer:
    def main(self):
        """Main function to run the gesture recognition"""
        num_hands = 1
        model_path = "custom_model_2.task"
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.lock = threading.Lock() 
        self.current_gestures = [] 

        options = GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands = num_hands,
            result_callback=self.__result_callback)
        recognizer = GestureRecognizer.create_from_options(options)

        timestamp = 0 
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=num_hands,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65)

        cap = cv2.VideoCapture(0)

        #a loop for the gesture recognition per frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            #converting the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #processing the frame to detect hands and landmarks
            results = hands.process(frame)

            #converting frame back to BGR for rendering
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            #code to recognize gestures if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    #drawing hand landmarks on the frame
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 139, 69), thickness=1, circle_radius=3))
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
                    recognizer.recognize_async(mp_image, timestamp)
                    timestamp = timestamp + 1 
               
                self.put_gestures(frame)
            cv2.imshow('Gesture Recognition using Mediapipe', frame)

            if cv2.waitKey(1) & 0xFF == 27:#breaking loop on pressing esc button
                break

    def put_gestures(self, frame):
        """Puts the name of the recognized gestures on the frame"""
        self.lock.acquire()
        gestures = self.current_gestures
        self.lock.release()
        y_pos = 50
        for hand_gesture_name in gestures:
            cv2.putText(frame, hand_gesture_name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (255,255,255), 2, cv2.LINE_AA)
            y_pos += 50

    def __result_callback(self, result, output_image, timestamp_ms):
        """Callback function to get the result of the gesture recognition"""
        #print(f'gesture recognition result: {result}')  #prints the whole result 
        self.lock.acquire() 
        self.current_gestures = []
        if result is not None and any(result.gestures):
            print("Recognized gestures:")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                print(gesture_name)
                if gesture_name == 'mute':
                    mute_audio() 
                elif gesture_name == 'ok':
                    unmute_audio()
                elif gesture_name == 'four':
                    set_full_volume()
                elif gesture_name == 'three':
                    set_half_volume()
                elif gesture_name == 'two_up_inverted':
                    set_qtr_volume()
                self.current_gestures.append(gesture_name)
        self.lock.release()

if __name__ == "__main__":
    rec = GestureRecognizer()
    rec.main()