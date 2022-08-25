import enum
from random import uniform
from formatting import csvToArray,intToOnehot
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential



POP_SIZE = 4
EPOCH_NUM = 5
EVOLUTION_STAGES = 5
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

def gaussianMutation():
    pass

def layersCrossover(chromosomes):
    children = []
    for i in range(int(POP_SIZE/2)):
        parentsIndexes = np.random.randint(0,len(chromosomes),2)
        father = chromosomes[parentsIndexes[0]]
        mother = chromosomes[parentsIndexes[1]]
        coinToss = np.random.randint(0,2)
        if coinToss == 0 :
            child = list([father[0],father[1],mother[2],mother[3]])
        else:
            child = list([mother[0],mother[1],father[2],father[3]])
        children.append(child)
    return children

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
            acc = model.evaluate(x_test,y_test)[1]
            chromosomeScores.append(acc)
        print(chromosomeScores)
        # save top half of chromosomes
        winningChromosomes = []
        for i in range(int(POP_SIZE/2)):
            bestIndex = np.argmax(chromosomeScores)
            winningChromosomes.append(chromosomeList[bestIndex])
            chromosomeScores[bestIndex]=0
        # Crossover
        children = layersCrossover(winningChromosomes)
        chromosomeList = winningChromosomes + children
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
    model.fit(x_train, y_train, epochs=EPOCH_NUM * EVOLUTION_STAGES * POP_SIZE, validation_data=(x_test,y_test))

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