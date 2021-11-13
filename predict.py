from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2


IMAGE_DIMS = (96, 96, 2)

mlb = pickle.loads(open("mlb.pkl", "rb").read())
model = load_model('model_fashion_multitask_learning.h5')
def _get_image(path):
  img = cv2.imread(path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def _predict_image(image, model, mlb):
  # Lấy kích thước 3 kênh của image
  (w, h, c) = image.shape
  # Nếu resize width = 400 thì height resize sẽ là
  height_rz = int(h*400/w)
  # Resize lại ảnh để hiện thị
  output = cv2.resize(image, (height_rz, 400))
  # Resize lại ảnh để dự báo
  image = cv2.resize(image, IMAGE_DIMS[:2])/255.0
  # Dự báo xác suất của ảnh
  prob = model.predict(np.expand_dims(image, axis=0))[0]
  # Trích ra 2 xác suất cao nhất
  argmax = np.argsort(prob)[::-1][:2]
  # Show classes và probability ra ảnh hiển thị
  for (i, j) in enumerate(argmax):
    # popup nhãn và xác suất dự báo lên ảnh hiển thị
    label = "{}: {:.2f}%".format(mlb.classes_[j], prob[j] * 100)
    cv2.putText(output, label, (5, (i * 20) + 15),
      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 0, 0), 2)
  # show the output image
  output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
  # plt.imshow(output)
  # cv2.imwrite('predict.jpg', output)
  cv2.imshow("Output", output)
  cv2.waitKey(0)

image = _get_image('predict_images/quannamjean.jpg')
_predict_image(image, model, mlb)