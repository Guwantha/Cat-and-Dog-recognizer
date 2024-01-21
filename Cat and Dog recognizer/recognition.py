import cv2

path = 'E:/Competitions/SLIOT - PawSitter/Cat and Dog recognizer/captured/pic.jpg'

thres = 0.6 # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames = []
classFile = 'coco.names'    # 17 - Cat, 18 - Dog
with open(classFile,'rt') as f:
    classNames = [line.rstrip() for line in f]


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            animalID = classId - 1
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[animalID],(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


    cv2.imshow('Output',img)
    
    for classId in classIds:
        if classId == 17:
           cv2.imwrite(path, img) 
           exec(open('predict_animal.py').read())

        elif classId == 18:
            cv2.imwrite(path, img)
            exec(open('predict_animal.py').read())

    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
