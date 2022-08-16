from formatting import csvToArray,intToOnehot

import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential



POP_SIZE = 10
EPOCH_NUM = 20

def geneticAlgoBasedTraining(x_train,y_train,x_test,y_test):
    model = Sequential()
    model.add(Flatten(input_shape=(3072,)))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
    # inisiate weights(chromosome) list
    for chromosome in chromosomeList:
        for i in range(EPOCH_NUM):
            # set_weights of model to chromosome
            model.train_on_batch(x_train, y_train)
            # save weights after SGD
    # test_on_batch for every chromosome
    # save top half of chromosomes
    # Crossover
    # Mutation
    # all over again
    weights=model.get_weights()
    #
def main():
    x_train,y_train = csvToArray("train.csv")
    x_test,y_test= csvToArray("validate.csv")

    y_train = intToOnehot(y_train)
    y_test = intToOnehot(y_test)

    model = Sequential()
    model.add(Flatten(input_shape=(3072,)))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
    # inisiate weights(chromosome) list
    for chromosome in chromosomeList:
        for i in range(EPOCH_NUM):
            # set_weights of model to chromosome
            model.train_on_batch(x_train, y_train)
            # save weights after SGD
    # test_on_batch for every chromosome
    # save top half of chromosomes
    # Crossover
    # Mutation
    # all over again
    weights=model.get_weights()
    #


    #model.fit(x_train, y_train, epochs=10, validation_data=(x_test,y_test))

if __name__ == '__main__':
    main()