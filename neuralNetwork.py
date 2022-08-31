
from random import uniform
from formatting import csvToArray,intToOnehot
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential

import random

POP_SIZE = 8
EPOCH_NUM = 50
EVOLUTION_STAGES = 50
MU,SIGMA = 0,0.5
GAUSSIAN_MUTATION_PROB = 0.4
def initiateChromosomeList(model):
    chromosomeList = []
    for chromosomeIndex in range(POP_SIZE):
    #    randomizedFirstWeight= default_rng().uniform(low= -1,high=1,size= (3072,32))
    #    randomizedFirstBias= default_rng().uniform(low= -1,high=1,size= (32,))
    #    randomizedSecondWeight =  default_rng().uniform(low= -1,high=1,size= (32,10))
    #    randomizedSecondBias = default_rng().uniform(low= -1,high=1,size= (10,))
        chromosome = model.get_weights()
        chromosomeList.append(chromosome)
    return chromosomeList

def gaussianMutation(chromosome):
    w1WeightsGaussianAddition = np.random.normal(MU,SIGMA,(3072,32))
    w1BiasGaussianAddition = np.random.normal(MU,SIGMA,(32,))
    w2WeightsGaussianAddition = np.random.normal(MU,SIGMA,(32,10))
    w2BiasGaussianAddition = np.random.normal(MU,SIGMA,(10,))
    chromosome[0] += w1WeightsGaussianAddition
    chromosome[1] += w1BiasGaussianAddition
    chromosome[2] += w2WeightsGaussianAddition
    chromosome[3] += w2BiasGaussianAddition
    return chromosome

def weightsCrossover(chromosomes):
    children = []
    for i in range(int(POP_SIZE/2)):
        child = [np.empty((3072,32)),np.empty((32,)),np.empty((32,10)),np.empty((10,))]
        parentsIndexes = random.sample(range(0, len(chromosomes)), 2)
        father = chromosomes[parentsIndexes[0]]
        mother = chromosomes[parentsIndexes[1]]

        # Child first layer w1 weights 
        for w1EnteringNeuronSetIndex,w1EnteringNeuronSet in enumerate(child[0].T):
            coinToss = np.random.randint(0,2)
            if coinToss == 0 :
                child[0].T[w1EnteringNeuronSetIndex] = father[0].T[w1EnteringNeuronSetIndex]
            else:
                child[0].T[w1EnteringNeuronSetIndex]= mother[0].T[w1EnteringNeuronSetIndex]
        # Child first layer w1 bias
        coinToss = np.random.randint(0,2)
        if coinToss == 0 :
            child[1] = father[1]
        else:
            child[1] = mother[1]
        # Child second layer w2 weights 
        for w2EnteringNeuronSetIndex,w2EnteringNeuronSet in enumerate(child[2].T):
            coinToss = np.random.randint(0,2)
            if coinToss == 0 :
                child[2].T[w2EnteringNeuronSetIndex] = father[2].T[w2EnteringNeuronSetIndex]
            else:
                child[2].T[w2EnteringNeuronSetIndex]= mother[2].T[w2EnteringNeuronSetIndex]
        # Child second layer w2 bias
                coinToss = np.random.randint(0,2)
        if coinToss == 0 :
            child[3] = father[3]
        else:
            child[3] = mother[3]
        # Add to children list
        children.append(child)
    
    return children

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
    
    chromosomeList = initiateChromosomeList(model)

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
        children = weightsCrossover(winningChromosomes)
        chromosomeList = winningChromosomes + children
        #savedWinningChromosomes = copy.deepcopy(winningChromosomes)
        # Mutation
        for index,chromosome in enumerate(chromosomeList):
            # Skip mutation on the first chromosome in list,which had the best score 
           if index==0:
            continue
           if np.random.random() < GAUSSIAN_MUTATION_PROB:
               
               chromosome= gaussianMutation(chromosome)
        
        
        
def regularTraining(x_train,y_train,x_test,y_test):
    model = Sequential()
    model.add(Flatten(input_shape=(3072,)))
    model.add(Dense(32, activation='sigmoid',name = 'w1'))
    model.add(Dense(10, activation='softmax',name='w2'))
    model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
    model.summary()
    model.fit(x_train, y_train, epochs= EPOCH_NUM, validation_data=(x_test,y_test))
    #EPOCH_NUM * EVOLUTION_STAGES
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