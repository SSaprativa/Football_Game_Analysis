# importing YOLO model
from ultralytics import YOLO

# using the model from the models folder that is downloaded
# we train the model in the 'training' folder and downloaded the trained model
# we do not use pretrained YOLO model because it has multiple problems such as-
## referees are also marked as a person which we do not want.
## we want to identify referees different and player diffrent
## another problem is pretrained YOLO model also identifying Person outside the field which we do not want.
# so we download roboflow pretrained data on football dataset that will identify referees as different and does not detect person on otside ground.
# we train our YOLO model based on the dataset and use it in our input video

# training Yolo MODEL
model = YOLO("models/best.pt")

# used the trained model on our input video
results = model.predict('input_videos/08fd33_4.mp4',save=True)

# print the result for the first frame
print(results[0])
print("======================")

# print the coordinate of the box
for box in results[0].boxes:
    print(box)