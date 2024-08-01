from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# define the LeNet-5 model?
def create_lenet_model(input_shape=(128, 128, 3), num_classes=2):
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# the figure size
IMSIZE = 128
input_shape = (IMSIZE, IMSIZE, 3)

#  develop LeNet-5 model
model = create_lenet_model(input_shape)

optimizer = Adam(lr=0.001)

# compile model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "E://table//gender//train",
    target_size=(IMSIZE, IMSIZE),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    "E://table//gender//test",
    target_size=(IMSIZE, IMSIZE),
    batch_size=32,
    class_mode='categorical'
)

# train model
num_epochs = 20
model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator)

# validation model
model.summary()
accuracy = model.evaluate(validation_generator)[1]
print(f"Validation Accuracy: {accuracy}")
