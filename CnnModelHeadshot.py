import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras import regularizers
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

train_dir = 'photo_test2/train'
test_dir = 'photo_test2/test'
validation_dir = 'photo_test2/validation'


model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                input_shape=(224,224,3),
                 activation='relu'
                ))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(filters=128,
                 kernel_size=(3,3),
                 activation='relu'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))


#fc
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_regularizer=regularizers.l2(l=0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.2))


model.add(Dense(1,activation='sigmoid'))


# 顯示出模型摘要
model.summary()

estop = EarlyStopping(monitor='val_loss', patience=5)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'] )

train_datagen =  ImageDataGenerator(
  rescale=1./255,
      zoom_range = 0.2,
      shear_range = 0.2, #指定將影象像素縮放到0~1之間
  )


test_datagen = ImageDataGenerator(rescale=1./255)


# 訓練資料與測試資料  #分類超過兩類 使用categorical, 若分類只有兩類使用binary
train_generator = train_datagen.flow_from_directory(
train_dir,
target_size=(224, 224),
batch_size=32,
class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
validation_dir,
target_size=(224, 224),
batch_size=32,
class_mode='binary',
)

# 使用批量生成器 模型模型
H = model.fit_generator(
train_generator,
steps_per_epoch=train_generator.samples/train_generator.batch_size, #一共訓練100次
epochs=32,
validation_data=validation_generator,
validation_steps=20,
callbacks=[estop]
)

model.save('model_CnnModel_m1_by.h5')

# 顯示acc學習結果
accuracy = H.history['acc']
val_accuracy = H.history['val_acc']
plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示loss學習結果
loss = H.history['loss']
val_loss = H.history['val_loss']
plt.plot(range(len(loss)), loss, marker='.', label='loss(training data)')
plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

num_of_test_samples = 1195
batch_size=32


Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred.shape)
print(validation_generator.classes.shape)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))












