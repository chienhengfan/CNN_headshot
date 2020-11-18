from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping
import numpy as np

# 資料準備
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,# ((x/255)-0.5)*2  歸一化到±1之間
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(directory='./photo_test3/train',
                                  target_size=(299,299),#Inception V3規定大小
                                  batch_size=64)
val_generator = val_datagen.flow_from_directory(directory='./photo_test3/validation',
                                target_size=(299,299),
                                batch_size=64)


base_model = InceptionV3(weights='imagenet',include_top=False)

# 增加新的輸出層
x = base_model.output
x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 將 MxNxC 的張量轉換成 1xC 張量，C是通道數
x = Dense(1024,activation='relu')(x)
predictions = Dense(3,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)
# plot_model(model,'tlmodel.png')


def setup_to_transfer_learning(model,base_model):#base_model
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 3 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001),loss='categorical_crossentropy',metrics=['acc'])


estop = EarlyStopping(monitor='val_loss', patience=5)
setup_to_transfer_learning(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=40,
                    epochs=15,
                    validation_data=val_generator,
                    validation_steps=12,
                    class_weight='auto'
                    )
model.save('./headshot_inceptionv3.h5')

# 訓練資料與測試資料  #分類超過兩類 使用categorical, 若分類只有兩類使用binary
train_generator = train_datagen.flow_from_directory(
'./photo_test3/train',
target_size=(224, 224),
batch_size=32,
class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
'./photo_test3/validation',
target_size=(224, 224),
batch_size=32,
class_mode='categorical',
)





accuracy = history_tl.history['acc']
val_accuracy = history_tl.history['val_acc']
plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示loss學習結果
loss = history_tl.history['loss']
val_loss = history_tl.history['val_loss']
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
