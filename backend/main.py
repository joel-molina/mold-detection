#website imports
from flask import Flask, render_template, request, jsonify
import numpy as np
import base64
from dotenv import load_dotenv

#display imports
from PIL import Image
import cv2

#model imports
import torch
from torchvision import transforms, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from model import Network #classification model


load_dotenv()
#define flask object, template (html), and static (css,js) 
app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

#define the html file to be run on the home page
@app.route("/")
def home():
    return render_template("index.html")

#initialize classification model
classification_model = Network()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model.to(device)
classification_model.load_state_dict(torch.load("model-weights/best_classifier.pth", weights_only=True, map_location=device))
classification_model.eval()

#initialize object detection model
detection_model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
num_classes=2
in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
detection_model.to(device)
detection_model.load_state_dict(torch.load("model-weights/best_detector.pth", weights_only=True, map_location=device))
detection_model.eval()

#define dynamic site update using uploaded image
@app.route("/predict", methods=["POST"])
def prediction():
    #get image
    file = request.files["file"]
    original_image = Image.open(file.stream)
    original_image = original_image.convert("RGB")

    #classification image preprocessing 
    transform = transforms.Compose([
    transforms.Resize((256,256)), 
    transforms.ToTensor()
    ])
    classification_image = transform(original_image)
    classification_image = classification_image.to(device)
    classification_image = classification_image.unsqueeze(0) #include batch size in tensor.

    #detection image preprocessing
    detection_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    detection_image = detection_transform(original_image)
    detection_image = detection_image.to(device)

    #classification prediction
    with torch.no_grad():
        prediction = classification_model(classification_image)
        prediction_label = torch.argmax(prediction, dim=1)

    #run object detection if mold detected.
    if prediction_label.item() == 1:
        response = "Bread mold detected."
        
        with torch.no_grad():
            pred = detection_model([detection_image])

        bboxes, scores = pred[0]["boxes"], pred[0]["scores"]

        #NMS Filtering
        keep = torch.where(scores > 0.1)[0]    
        nms_indices = nms(bboxes[keep], scores[keep], 0.2)

        #final target data
        bboxes = bboxes[keep][nms_indices]
        #print(bboxes)
        #print(scores)

        
        #convert tensor to np array and adjust colormap (opencv uses BGR)
        output_image = np.array(detection_image.cpu().permute(1, 2, 0).numpy() * 255, dtype=np.uint8)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        #rescale image AND bounding boxes
        height, width, _ = output_image.shape
        output_image = cv2.resize(output_image, (500,500))
        scale_x = 500 / width
        scale_y = 500 / height

        #draw bounding boxes
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(bboxes)):
            x1,y1,x2,y2 = bboxes[i].cpu().numpy().astype("int")

            x1 = int(x1*scale_x)
            y1 = int(y1*scale_y)
            x2 = int(x2*scale_x)
            y2 = int(y2*scale_y)

            class_name = "Mold"
            output_image = cv2.rectangle(output_image, (x1, y1), (x2, y2), (52, 137, 235), 3)
            output_image = cv2.putText(output_image, class_name, (x1, y1 - 10), font, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

        #convert numpy array to jpeg then to base64 to send to website.
        _, buffer = cv2.imencode(".jpg", output_image)
        output_image_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"result": response, "image": output_image_b64})
    else:
        response = "No bread mold detected."
    return jsonify({"result": response})

#run app
if __name__ == "__main__":
    app.run()

