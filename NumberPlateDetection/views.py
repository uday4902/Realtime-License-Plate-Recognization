from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
import time

def button(request):
    os.remove("media\\img\\images.jpg")
    return render(request,'index.html')

def live(request):
    model_cfg_path = os.path.join('.','NumberPlateDetection', 'model', 'cfg', 'darknet-yolov3.cfg')
    model_weights_path = os.path.join('.','NumberPlateDetection', 'model', 'weights', 'model.weights')
    class_names_path = os.path.join('.', 'NumberPlateDetection','model', 'class.names')
    cam=cv2.VideoCapture(0)
    while True:
        ret,frame=cam.read()
        if not ret:
            print("failed to captur the image")
            break
        cv2.imshow("test",frame)
        k=cv2.waitKey(1)
        if k%256==27:
            print("closed")
            break
        elif k%256==32:
            img_name="media/image.jpg"
            cv2.imwrite(img_name,frame)
            print("image taken")
    cam.release()
    cv2.destroyAllWindows()
    if 'img_name' in locals():
        tmp_file = os.path.join(settings.MEDIA_ROOT,"image.jpg")
        extracted_text,plate_detected,confidence=algo(model_cfg_path,model_weights_path,class_names_path,tmp_file)
        if plate_detected and confidence>0.5:
            message=extracted_text
        elif plate_detected:
            message = f"License plate possibly detected with low confidence: {extracted_text}"
        else:
            message="License Plate is not detected"

        context={'extracted_text':message, 'plate_detected':plate_detected, 'timestamp':int(time.time())}
    else:
        context={'extracted_text':'no image is extracted', 'plate_detected': False}
    return render(request,"index.html", context)
    

def output(request):
    os.remove("media\\tmp\\image.jpg")
    model_cfg_path = os.path.join('.','NumberPlateDetection', 'model', 'cfg', 'darknet-yolov3.cfg')
    model_weights_path = os.path.join('.','NumberPlateDetection', 'model', 'weights', 'model.weights')
    class_names_path = os.path.join('.', 'NumberPlateDetection','model', 'class.names')
    if request.method == "POST" :
        formimage = request.FILES['image']
        path = default_storage.save('tmp\image.jpg',ContentFile(formimage.read()))
        print(path)
        tmp_file = os.path.join(settings.MEDIA_ROOT,path)
        print(tmp_file)
        extracted_text,plate_detected,confidence=algo(model_cfg_path,model_weights_path,class_names_path,tmp_file)
        context={'extracted_text':extracted_text,'plate_detected':plate_detected,'confidence_score':confidence}
    return render(request,"index.html", context)

def algo(model_cfg_path,model_weights_path,class_names_path,tmp_file):
    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()
    license_plate=None
    confidence_score=0.0
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)
        
    img = cv2.imread(tmp_file)

    H, W, _ = img.shape

    # convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
            # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]
        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
        bbox_confidence = detection[4]
        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])
        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)
        confidence_score=max(confidence_score,score)
        # apply nms
    bboxes, class_ids, scores = NMS(bboxes, class_ids, scores)

        # plot
    plate_detected=False
    plate_text=[]
    reader = easyocr.Reader(['en'])
    for bbox_, bbox in enumerate(bboxes):
        plate_detected=True
        xc, yc, w, h = bbox

        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

        img = cv2.rectangle(img,(int(xc - (w / 2)), int(yc - (h / 2))),(int(xc + (w / 2)), int(yc + (h / 2))),(0, 255, 0),15)

        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY) 
        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
        output = reader.readtext(license_plate_thresh)

        for out in output:
            text_bbox, text, text_score = out
            if text_score > 0.4:
                print(text, text_score)
                plate_text.append(text)
    if license_plate is not None:
        cv2.imwrite('media/img/images.jpg',license_plate)
    else:
        print('License Plate is not detected')
    if not plate_detected:
        return "License Plate is not detected", False, confidence_score
    elif not plate_text:
        return "License Plate is detected but no text is extracted", True, confidence_score
    else:
        return ' '.join(plate_text),True, confidence_score
    
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))
    # plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))
    # plt.show()
def NMS(boxes, class_ids, confidences, overlapThresh = 0.5):

    boxes = np.asarray(boxes)
    class_ids = np.asarray(class_ids)
    confidences = np.asarray(confidences)

    # Return empty lists, if no boxes given
    if len(boxes) == 0:
        return [], [], []

    x1 = boxes[:, 0] - (boxes[:, 2] / 2)  # x coordinate of the top-left corner
    y1 = boxes[:, 1] - (boxes[:, 3] / 2)  # y coordinate of the top-left corner
    x2 = boxes[:, 0] + (boxes[:, 2] / 2)  # x coordinate of the bottom-right corner
    y2 = boxes[:, 1] + (boxes[:, 3] / 2)  # y coordinate of the bottom-right corner

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):
        # Create temporary indices
        temp_indices = indices[indices != i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0] - (box[2] / 2), boxes[temp_indices, 0] - (boxes[temp_indices, 2] / 2))
        yy1 = np.maximum(box[1] - (box[3] / 2), boxes[temp_indices, 1] - (boxes[temp_indices, 3] / 2))
        xx2 = np.minimum(box[0] + (box[2] / 2), boxes[temp_indices, 0] + (boxes[temp_indices, 2] / 2))
        yy2 = np.minimum(box[1] + (box[3] / 2), boxes[temp_indices, 1] + (boxes[temp_indices, 3] / 2))

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if overlapping greater than our threshold, remove the bounding box
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]

    # return only the boxes at the remaining indices
    return boxes[indices], class_ids[indices], confidences[indices]


def get_outputs(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    outs = net.forward(output_layers)

    outs = [c for out in outs for c in out if c[4] > 0.1]

    return outs


def draw(bbox, img):

    xc, yc, w, h = bbox
    img = cv2.rectangle(img,
                        (xc - int(w / 2), yc - int(h / 2)),
                        (xc + int(w / 2), yc + int(h / 2)),
                        (0, 255, 0), 20)

    return img
