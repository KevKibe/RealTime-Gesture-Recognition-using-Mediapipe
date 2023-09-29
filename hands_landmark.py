import cv2
import mediapipe as mp
import math

class GestureRecognizer:
    def __init__(self):
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


    
    #modify this method to detect any hand signal you want eg detecting thumbs down, thumbs up etc 
    def detect_peace_sign(self, hand_landmarks):
        # Get the tip of the middle finger, index finger, and wrist
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Calculate the distances between the middle finger and index finger tips and the wrist
        distance_middle_to_wrist = math.sqrt((middle_finger_tip.x - wrist.x)**2 + (middle_finger_tip.y - wrist.y)**2)
        distance_index_to_wrist = math.sqrt((index_finger_tip.x - wrist.x)**2 + (index_finger_tip.y - wrist.y)**2)

        # Check if the middle finger and index finger are sufficiently extended
        finger_extension_threshold = 1.0  # Adjust this threshold based on testing
        if distance_middle_to_wrist > finger_extension_threshold and distance_index_to_wrist > finger_extension_threshold:
            # Calculate the angle between the fingers and the wrist using dot product
            dot_product = (middle_finger_tip.x - wrist.x) * (index_finger_tip.x - wrist.x) + (middle_finger_tip.y - wrist.y) * (index_finger_tip.y - wrist.y)
            mag_middle = math.sqrt((middle_finger_tip.x - wrist.x)**2 + (middle_finger_tip.y - wrist.y)**2)
            mag_index = math.sqrt((index_finger_tip.x - wrist.x)**2 + (index_finger_tip.y - wrist.y)**2)
            cos_angle = dot_product / (mag_middle * mag_index)

            # Calculate the angle in degrees
            angle = math.degrees(math.acos(cos_angle))

            # Define the angle range for the peace sign
            peace_sign_angle_range = (75, 100)  # Adjust this range based on testing

            # Check if the angle is within the peace sign angle range
            if peace_sign_angle_range[0] < angle < peace_sign_angle_range[1]:
                return True

  
    def detect_thumbsup(self,hand_landmarks):
        # Get the tip of the thumb and the tip of the index finger
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Calculate the Euclidean distance between the thumb tip and the index finger tip
        distance = ((thumb_tip.x - index_finger_tip.x)**2 + (thumb_tip.y - index_finger_tip.y)**2)**0.5

        # Define a threshold to consider it a thumbs-up gesture
        thumb_up_threshold = 0.08

        # Check if the distance is below the threshold (thumbs-up)
        if distance < thumb_up_threshold:
            return True
        else:
            return False

    #modify this method to detect any hand signal you want eg detecting thumbs down, thumbs up etc 
    #this method returns ngumi mbwegze print statement
    def detect_two_fists(self, hand_landmarks_list):
        # Check if two hands are detected
        if hand_landmarks_list == 2:
            # Define a threshold for the distance between finger tips and the wrist
            fist_distance_threshold = 0.5  # Adjust this threshold based on testing

            # Check each hand for a fist
            for hand_landmarks in hand_landmarks_list:
                middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

                # Calculate the distance between the middle finger tip and the wrist
                distance_middle_to_wrist = math.sqrt((middle_finger_tip.x - wrist.x)**2 + (middle_finger_tip.y - wrist.y)**2)

                # Calculate the distance between the index finger tip and the wrist
                distance_index_to_wrist = math.sqrt((index_finger_tip.x - wrist.x)**2 + (index_finger_tip.y - wrist.y)**2)

                # Check if both fingers are close to the wrist (forming a fist)
                if distance_middle_to_wrist < fist_distance_threshold and distance_index_to_wrist < fist_distance_threshold:
                    return True

        return False



    #this proceses the video frames for gesture recognition
    def process_video(self):
        cap = cv2.VideoCapture(0)#starts video feed 

        while True:
            ret, frame = cap.read()                               #reads frames in video feed
            frame = cv2.flip(frame, 1)                            #flips frames horizontally
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    #converts frame from BGR to RGB

            frame_rgb.flags.writeable = False                     #this improves performance in Mediapipe
            result = self.hands.process(frame_rgb)                #processing frame to detect hand landmarks
            frame_rgb.flags.writeable = True

            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)     #converting frame to BGR for rendering
            
            #drawing landmarks on the frame if hands are detected
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                                self.mp_draw.DrawingSpec(color=(255, 0, 255),thickness=5, circle_radius=5))
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, connection_drawing_spec=self.mp_draw.DrawingSpec((0, 255, 0), thickness=5, circle_radius=3))
                    
                    # Detect the gesture and display the text output                        
                    if self.detect_peace_sign(hand_landmarks):
                        #text in frame if gesture is detected
                        cv2.putText(frame, "Peace..out", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    if self.detect_thumbsup(hand_landmarks):
                        #text in frame if gesture is detected
                        cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    if self.detect_two_fists(hand_landmarks):
                        #text in frame if gesture is detected
                        cv2.putText(frame, "Ngumi Mbwegze", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Peace Sign Hands Detection ", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.process_video()









