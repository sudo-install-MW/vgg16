from keras.layers import Dense
from keras.models import Sequential

# script to hold the CNN network

class network():
    def __init__(self):
        pass
    def fc_network(self, input_img, input_label, test_img, test_label):
        input_shape = input_img.shape[1]
        print("Input shape is", input_shape)
        # as first layer in a sequential model:
        model = Sequential()
        # layer 1

        model.add(Dense(784, input_dim=input_shape, activation='relu'))
        # layer 2
        model.add(Dense(784, activation='relu'))
        # layer 3
        model.add(Dense(10, activation='relu'))
        model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        model.fit(input_img, input_label, batch_size=4, nb_epoch=1, verbose=1)
        scores = model.evaluate(test_img, test_label)
        print("model accuracy in test set is :", scores)





