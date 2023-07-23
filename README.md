## Description
- This is a project that showcases finetuning a model and performing gesture recognition of 21 different gestures using Mediapipe from Google.
- The [notebook](https://github.com/KevKibe/Gesture-Recognition-using-Mediapipe/blob/main/finetuning_handrecognition_model.ipynb) shows how I trained the baseline model that achieved 83% accuracy and two finetuned models that achieved 88% accuracy all on the test set.
- The file [gesture_recognition.py](https://github.com/KevKibe/Gesture-Recognition-using-Mediapipe/blob/main/gesture_recognition.py) contains the code base to put the models to use using a live webcam feed. Scroll below to the usage section.

## Dataset
- The dataset is a combination of two datasets and you can get it [here](https://drive.google.com/file/d/1ILwgfolCd6Z6ar0WiDld_h-lYKYicWNk/view?usp=sharing). <br>
  **A sample of the data in the dataset**<br>
![image](https://github.com/KevKibe/Rock-Paper-Scissors-using-Mediapipe/assets/86055894/60ffc9ba-0fbc-4836-8f1e-7dd5d7674090)
![image](https://github.com/KevKibe/Rock-Paper-Scissors-using-Mediapipe/assets/86055894/5059fec9-f950-4127-91b2-f52d6bbf9c05)
![image](https://github.com/KevKibe/Rock-Paper-Scissors-using-Mediapipe/assets/86055894/e53f73e9-93c8-436e-8792-ff4cadc4c377)

## Installation
- Clone the repository: `git clone https://github.com/KevKibe/Gesture-Recognition-using-Mediapipe.git`
- Install dependencies: `pip install -r requirements.txt`

## Usage
- Assign the variable model path to one of the models. For example `model_path = "custom_model_2.task"`.
- Run the application by running the command `python gesture_recognition.py` in the terminal.
- To close the application press the ESC key.
  

