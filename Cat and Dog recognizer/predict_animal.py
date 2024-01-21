import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2 as cv

new_model = tf.keras.models.load_model(('pet_trained.h5'), custom_objects={'KerasLayer':hub.KerasLayer})
print ('Model Loaded')

def transform(img):
    resize = cv.resize(img, (224,224))
    rescale = resize/255
    reshape = np.reshape(rescale, [1,224,224,3])
    return reshape

def predict(reshape):
    prediction = new_model.predict(reshape)
    predict_label = np.argmax(prediction)
    return predict_label

path = 'E:/Competitions/SLIOT - PawSitter/Cat and Dog recognizer/captured/pic.jpg'

img = cv.imread(path)
cv.imshow('picture', img)

reshape = transform(img)

predict_label = predict(reshape)


if predict_label == 0:
    print('Cat')
else:   
    print('Dog')




# capture = cv.VideoCapture(0)
# while True:
#     isTrue, frame = capture.read()
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     predict_label = predict(frame)
#     print(predict_label)
#     # if predict_label == 0:
#     #     print('Cat')
#     # else:   
#     #     print('Dog')

#     cv.imshow('Deteted Face', frame)

#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break

# if predict_label == 0:
#     print('Cat')
# else:   
#     print('Dog')

# capture.release()
# cv.destroyAllWindows()