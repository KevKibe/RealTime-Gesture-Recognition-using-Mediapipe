import cv2
import mediapipe as mp
from mediapipe.tasks import python
import threading 

class GestureRecognizer:
    def main(self):
        num_hands = 2
        model_path = "gesture_recognizer.task"
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


        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            np_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_array)
                    recognizer.recognize_async(mp_image, timestamp)
                    timestamp = timestamp + 1 # should be monotonically increasing, because in LIVE_STREAM mode
                    
                self.put_gestures(frame)

            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    def put_gestures(self, frame):
        self.lock.acquire()
        gestures = self.current_gestures
        self.lock.release()
        y_pos = 50
        for hand_gesture_name in gestures:
            # show the prediction on the frame
            cv2.putText(frame, hand_gesture_name, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0,0,255), 2, cv2.LINE_AA)
            y_pos += 50

    def __result_callback(self, result, output_image, timestamp_ms):
        #print(f'gesture recognition result: {result}')  #print the whole result 
        self.lock.acquire() # solves potential concurrency issues
        self.current_gestures = []
        if result is not None and any(result.gestures):
            print("Recognized gestures:")
            for single_hand_gesture_data in result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                print(gesture_name)
                self.current_gestures.append(gesture_name)
        self.lock.release()

if __name__ == "__main__":
    rec = GestureRecognizer()
    rec.main()
