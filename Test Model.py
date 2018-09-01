import cv2
import keras
import imutils
import numpy as np
from skimage.transform import resize
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.utils.generic_utils import CustomObjectScope

path = "C:\\Users\\Ibrahim\\Desktop\\Bone Detection\\DNN X-Ray 2 classess\\"
OUT_DIM = (96, 96)

model = load_model(path + "final_model.h5")

image = cv2.imread(path + "Test\\Edema.png")
orig = image.copy()
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

abnormal, normal = model.predict(image)[0]

label = "NORMAL" if normal > abnormal else "ABNROMAL"
print("{}".format(label))

output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("X-ray", output)
cv2.waitKey(0)