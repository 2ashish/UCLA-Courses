from collections import defaultdict

def getPosterior(priorOfA, priorOfB, likelihood):
    keysA = [k for k,v in priorOfA.items()]
    keysB = [k for k,v in priorOfB.items()]
    
    marginalLikelihoodOfA = defaultdict(lambda:0) #P(D/A)
    marginalLikelihoodOfB = defaultdict(lambda:0) #P(D/B)
    marginalOfA = {} #P(A/D)
    marginalOfB = {} #P(B/D)
    
    for k,v in likelihood.items():
        marginalLikelihoodOfA[k[0]]+=v*priorOfB[k[1]]
        marginalLikelihoodOfB[k[1]]+=v*priorOfA[k[0]]
    
    for k1 in keysA:
        marginalOfA[k1] = 0
        for k2 in keysB:
            marginalOfA[k1] += (priorOfA[k1]*priorOfB[k2]*likelihood[(k1,k2)])/marginalLikelihoodOfB[k2] #P(A=a/D) = sum_b{P(A=a)P(B=b)P(D/A=a,B=b)/P(D/B=b)}
    for k2 in keysB:
        marginalOfB[k2] = 0
        for k1 in keysA:
            marginalOfB[k2] += (priorOfA[k1]*priorOfB[k2]*likelihood[(k1,k2)])/marginalLikelihoodOfA[k1] #P(B=b/D) = sum_a{P(A=a)P(B=b)P(D/A=a,B=b)/P(D/A=a)}
    
    
    return([marginalOfA, marginalOfB])



def main():
    exampleOnePriorofA = {'a0': .5, 'a1': .5}
    exampleOnePriorofB = {'b0': .25, 'b1': .75}
    exampleOneLikelihood = {('a0', 'b0'): 0.42, ('a0', 'b1'): 0.12, ('a1', 'b0'): 0.07, ('a1', 'b1'): 0.02}
    print(getPosterior(exampleOnePriorofA, exampleOnePriorofB, exampleOneLikelihood))

    exampleTwoPriorofA = {'red': 1/10 , 'blue': 4/10, 'green': 2/10, 'purple': 3/10}
    exampleTwoPriorofB = {'x': 1/5, 'y': 2/5, 'z': 2/5}
    exampleTwoLikelihood = {('red', 'x'): 0.2, ('red', 'y'): 0.3, ('red', 'z'): 0.4, ('blue', 'x'): 0.08, ('blue', 'y'): 0.12, ('blue', 'z'): 0.16, ('green', 'x'): 0.24, ('green', 'y'): 0.36, ('green', 'z'): 0.48, ('purple', 'x'): 0.32, ('purple', 'y'): 0.48, ('purple', 'z'): 0.64}
    print(getPosterior(exampleTwoPriorofA, exampleTwoPriorofB, exampleTwoLikelihood))




if __name__ == '__main__':
    main()
