import EvGenLab as eg
import random as rand
import Chase as ch

def mutateCrossover(ind):
    i = rand.randint(0, 2)
    if i == 0:
        ind.vec[1] = eg.crossoverInsert
        ind.cross = "ins"
    elif i == 1:
        ind.vec[1] = eg.crossoverReflect
        ind.name = "ref"

MAX_MUTATION = 0.99
MIN_MUTATION = 0
MAX_HDIFF = 5
MIN_HDIFF = 0.1
MIN_LDIFF = 0.01
WAYS = []
PARANTS = eg.getParantsFight
PARANTSKW = {"adapt": ch.adaptationBrok}

class setIndiv(eg.individuel):
    # [population factor, mutation chance, crossover, hDiff, lDiff]
    def __init__(self):
        self.vec = [0.01, eg.crossoverInsert, 1.5, 0.5]
        self.cross = "ins"

    def fillRand(self):
        self.vec[0] = rand.uniform(MIN_MUTATION, MAX_MUTATION)
        mutateCrossover(self)
        self.vec[2] = rand.uniform(MIN_HDIFF, MAX_HDIFF)
        self.vec[3] = rand.uniform(MIN_LDIFF, self.vec[2])
        return self
    
    def insert(self, add, start, end=None):
        start %= len(self.vec)
        if end:
            end %= len(self.vec)
        if end and start > end:
            start, end = end, start
        if end and end > len(self.vec):
            end = len(self.vec)
        for i in range(start, end):
            self.vec[i] = add[i - start]
    
    def getName(self):
        s = str(self.vec[0])+self.cross+str(self.vec[2])+str(self.vec[3])
        return s

def mutation(indiv):
    r = rand.randint(1, 4)
    if r == 1:
        indiv.vec[0] = rand.uniform(MIN_MUTATION, MAX_MUTATION)
    elif r == 2:
        mutateCrossover(indiv)
    elif r == 3:
        indiv.vec[2] = rand.uniform(MIN_HDIFF, MAX_HDIFF)
    elif r == 4:
        indiv.vec[3] = rand.uniform(MIN_LDIFF, indiv.vec[2])

def runInd(ind):
    print("\tcalculating adaptation ...")
    ans = ch.runChase(
        size=len(WAYS[0]),
        populationSize=300,
        mutationRate=ind.vec[0],
        crossover=ind.vec[1],
        getParents=PARANTS,
        getParentsKw=PARANTSKW,
        mutation=eg.mutationSaltation,
        selection = eg.selectionSpinThatWheeel,
        steps = 60,
        hDiff = ind.vec[2],
        lDiff = ind.vec[3]
    )
    print("\tcalculating done")
    return ans

@eg.static_variables(res = dict())
def adaptation(ind):
    name = ind.getName()
    if name not in adaptation.res:
        answers =  []
        for _ in range(2):
            output = runInd(ind)
            answers.append(output[1])
        best = max(answers)
        adaptation.res[name] = 100 / best
    return adaptation.res[name] 


if __name__ == "__main__":
    with open("for_com.txt") as file:
        WAYS = [[float(i) for i in line.strip().split()] for line in file]
    ch.WAYS = WAYS
    alg = eg.GeneticCommivAlg(name="setUp", size=5, populationSize=10, mutationRateGen=ch.mutateRate, indiv=setIndiv)
    setUpParantsKW = {"adapt" : adaptation}
    setUpSelectionKW = {"size" : 10}
    for i in range(100):
        print(f"step {i}")
        alg.evolve(
            adaptation=adaptation, 
            getParents=PARANTS,
            crossover=eg.crossoverInsert, 
            mutation=mutation,
            selection = eg.selectionSpinThatWheeel,
            getParentsKwargs = setUpParantsKW,
            selectionKwargs=setUpSelectionKW)
        ans = max(alg.population, key=lambda a: adaptation(a))
        print(f"answer for now is {ans}")
    ans = max(alg.population, key=lambda a: adaptation(a))
    print(ans)