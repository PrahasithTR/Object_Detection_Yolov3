import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
glasses = []
with open("coco.names","r") as f:
    glasses = [line.strip() for line in f.readlines()]
print(glasses)
l_names = net.getLayerNames()
outputlayers = [l_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(glasses), 3))
# Reading image
image = cv2.imread("image1.jpg")
#changing the size
image = cv2.resize(image,None,fx=0.7,fy=0.7)
height,width,channels = image.shape

#Detecting objects
blob = cv2.dnn.blobFromImage(image,0.00392,(416,416),(0,0,0),True)

'''for b in blob:
    for n,image_b in enumerate(b):
        cv2.imshow(str(n),image_b)'''

net.setInput(blob)

outs = net.forward(outputlayers)
#print(outs)

#Showing information on the screen
cla_ids=[]
confi = []
box = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence= scores[class_id]
        if confidence > 0.5:
            #Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            #cv2.circle(image,(center_x,center_y),10,(0,255,0),2)
            x = int(center_x-w/2)
            y = int(center_y-h/2)

            #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            box.append([x,y,w,h])
            confi.append(float(confidence))
            cla_ids.append(class_id)

print(len(box))
indexes = cv2.dnn.NMSbox(box,confi,0.5,0.4)
print(indexes)
font = cv2.FONT_HERSHEY_SIMPLEX
number_objects_detected = len(box)
for i in range(len(box)):
    if i in indexes:
        x,y,w,h, = box[i]
        label = str(glasses[cla_ids[i]])
        color = colors[i]
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        cv2.putText(image, label, (x,y+30),font,0.5,color,2)
    
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
