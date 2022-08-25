import enum
from random import uniform
from formatting import csvToArray,intToOnehot
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential



POP_SIZE = 2
EPOCH_NUM = 10
EVOLUTION_STAGES = 10

def initiateChromosomeList():
    chromosomeList = []
    for chromosomeIndex in range(POP_SIZE):
        randomizedFirstWeight= default_rng().uniform(low= -1,high=1,size= (3072,32))
        randomizedFirstBias= default_rng().uniform(low= -1,high=1,size= (32,))
        randomizedSecondWeight =  default_rng().uniform(low= -1,high=1,size= (32,10))
        randomizedSecondBias = default_rng().uniform(low= -1,high=1,size= (10,))
        chromosome = list([randomizedFirstWeight,randomizedFirstBias,randomizedSecondWeight,randomizedSecondBias])
        chromosomeList.append(chromosome)
    return chromosomeList


def geneticAlgoBasedTraining(x_train,y_train,x_test,y_test):
    model = Sequential()
    model.add(Flatten(input_shape=(3072,)))
    model.add(Dense(32, activation='sigmoid',name = 'w1'))
    model.add(Dense(10, activation='softmax',name='w2'))
    model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
    model.summary()
    
    chromosomeList = initiateChromosomeList()

    for stage in range(EVOLUTION_STAGES):
        for chromosomeIndex,chromosome in enumerate(chromosomeList):
            model.get_layer('w1').set_weights([chromosome[0],chromosome[1]])
            model.get_layer('w2').set_weights([chromosome[2],chromosome[3]])
            model.fit(x_train, y_train,epochs=EPOCH_NUM,validation_data=(x_test,y_test))
            chromosomeList[chromosomeIndex] = model.get_weights()

        # test_on_batch for every chromosome
        chromosomeScores = []
        for chromosomeIndex,chromosome in enumerate(chromosomeList):
            model.get_layer('w1').set_weights([chromosome[0],chromosome[1]])
            model.get_layer('w2').set_weights([chromosome[2],chromosome[3]])
            acc = model.evaluate(x_train,y_train)[1]
            chromosomeScores.append(acc)
        print(chromosomeScores)
        # save top half of chromosomes
        winningChromosomes = []
        for i in range(int(POP_SIZE/2)):
            bestIndex = np.argmax(chromosomeScores)
            winningChromosomes.append(chromosomeList[bestIndex])
            chromosomeScores[bestIndex]=0
        # Crossover

        # Mutation
        # all over again
        
        
def regularTraining(x_train,y_train,x_test,y_test):
    model = Sequential()
    model.add(Flatten(input_shape=(3072,)))
    model.add(Dense(32, activation='sigmoid',name = 'w1'))
    model.add(Dense(10, activation='softmax',name='w2'))
    model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
    model.summary()
    model.fit(x_train, y_train, epochs=EPOCH_NUM, validation_data=(x_test,y_test))

    pass

def main():
    x_train,y_train = csvToArray("train.csv")
    x_test,y_test= csvToArray("validate.csv")
    y_train = intToOnehot(y_train)
    y_test = intToOnehot(y_test)
    regularTraining(x_train,y_train,x_test,y_test)
    geneticAlgoBasedTraining(x_train,y_train,x_test,y_test)
    return

if __name__ == '__main__':
    main()