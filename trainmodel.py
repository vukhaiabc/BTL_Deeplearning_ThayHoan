import glob2
import pandas as pd
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from net.modelMuti import KhaiPtitNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


def _list_images(root_dir, exts = ['.jpg', '.jpeg', '.png']):
  list_images = glob2.glob('dataset'+'/**')
  image_links = []
  for image_link in list_images:
    for ext in exts:
      if ext in image_link[-5:]:
        image_links.append(image_link)
  return image_links

imagePaths = sorted(_list_images(root_dir='dataset'))
print(imagePaths)
labels = [path.split("\\")[1] for path in imagePaths]
print(labels)
data = pd.DataFrame({'label': labels, 'source': imagePaths})
data.groupby('label').source.count().plot.bar()
# plt.show()
# build model :
INPUT_SHAPE = (96, 96, 3)
N_CLASSES = 6

model = KhaiPtitNet.build_model(inputShape=INPUT_SHAPE, classes=N_CLASSES,  finAct='sigmoid')
model.summary()
image_aug = ImageDataGenerator(rotation_range=25,
                         width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2,
	                       horizontal_flip=True, fill_mode="nearest")

LR_RATE = 0.01
EPOCHS = 50
opt = Adam(learning_rate=LR_RATE, decay=LR_RATE / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# convert anh and label :
images = []
labels = []
# Lấy list imagePaths theo từng label
data_sources = data.groupby('label').source.apply(lambda x: list(x))
# data_sources = data_sources[data_sources.index != 'blue_shirt']
for i, sources in enumerate(data_sources):
    np.random.shuffle(list(sources))
    # sources_200 = sources[:200]
    label = data_sources.index[i]
    sources = data_sources[label]
    for imagePath in sources:
        # Đọc dữ liệu ảnh
        image = cv2.imread(imagePath)
        image = cv2.resize(image, INPUT_SHAPE[:2])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        images.append(image)
        # Gán dữ liệu label
        fashion, color = label.split('_')
        labels.append([fashion, color])

# Stack list numpy array của ảnh thành một array
images = np.stack(images)
images = images / 255.0


mlb = MultiLabelBinarizer()
# One-hot encoding cho fashion
y = mlb.fit_transform(labels)

# Lưu trữ mlb.pkl file
f = open('mlb.pkl', "wb")
f.write(pickle.dumps(mlb))
f.close()
print('classes of labels: ', mlb.classes_)
print(y[1])

# train:
(X_train, X_val, y_train, y_val) = train_test_split(images, y,
                                                    test_size=0.2, random_state=123)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

BATCH_SIZE = 32
history = model.fit(
	image_aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
	validation_data=(X_val, y_val),
	steps_per_epoch=len(X_train) // BATCH_SIZE,
	epochs=EPOCHS, verbose=1)

model.save('model_fashion_multitask_learning.h5')