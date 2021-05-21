import random
from copy import deepcopy
import pandas as pd
from matplotlib import pyplot as plt


def static_variables(**kwargs):
    def decorator(fn):
        for k, v in kwargs.items():
            setattr(fn, k, v)
        return fn

    return decorator


class individuel:
    def __init__(self, size, maxEl=None):
        if maxEl == None:
            maxEl = size - 1
        self.vec = [0] * size
        self.max = maxEl

    def fill(self, vector):
        j = len(self.vec) - 1
        for i in range(len(vector) - 1, -1, -1):
            self.vec[j] = vector[i]
            j -= 1
            if j < 0:
                break
        return self

    def fillRand(self, bound=0):
        for i in range(len(self.vec)):
            self.vec[i] = random.randint(0, self.max)
        return self

    def __add__(self, num):
        left = num
        i = len(self.vec) - 1
        while left > 0:
            self.vec[i] += left
            left = self.vec[i] // (self.max + 1)
            self.vec[i] = self.vec[i] - (self.max + 1) * left
            i -= 1
            if i < 0:
                break
        return self

    def __pow__(self, vector):
        ans = deepcopy(self)
        for i in range(len(self.vec)):
            ans.vec[i] = (vector.vec[i] + self.vec[i]) % (self.max + 1)
        return ans

    def __ipow__(self, vector):
        for i in range(len(self.vec)):
            self.vec[i] = (vector.vec[i] + self.vec[i]) % (self.max + 1)
        return self

    def __irshift__(self, other):
        for i in range(len(self.vec) - 1, other - 1, -1):
            self.vec[i] = self.vec[i - other]
        for i in range(other):
            self.vec[i] = 0
        return self

    def __rshift__(self, other):
        ans = deepcopy(self)
        for i in range(len(self.vec) - 1, other - 1, -1):
            ans.vec[i] = self.vec[i - other]
        for i in range(other):
            ans.vec[i] = 0
        return ans

    def __repr__(self):
        return repr(self.vec)

    def insert(self, add, start, end=None):
        start %= len(self.vec)
        if end:
            end %= len(self.vec)
        if end and start > end:
            start, end = end, start
        for i in range(len(add)):
            self.vec[start + i] = add[i]
            if end and start + i:
                break
        return self


class uniqueIndiv(individuel):
    def __init__(self, size, maxEl=None, minEl=0):
        if not maxEl:
            maxEl = size - 1
        if maxEl - minEl < size:
            maxEl = minEl + size - 1
        self.vec = [i for i in range(size)]
        self.max = maxEl
        self.min = minEl

    def fillRand(self):
        elementList = [i for i in range(self.min, self.max + 1)]
        for i in range(len(self.vec)):
            self.vec[i] = random.choice(elementList)
            elementList.remove(self.vec[i])
        return self

    def insert(self, add, start, end=None):
        # breakpoint()
        start %= len(self.vec)
        if end:
            end %= len(self.vec)
        if end and start > end:
            start, end = end, start
        if end and end > len(self.vec):
            end = len(self.vec)
        copy = [self.min - 1] * len(self.vec)
        for i in range(len(add)):
            copy[start + i] = add[i]
            if end and start + i == end:
                break
        endInd = 0
        if end:
            endInd = end
        else:
            endInd = len(self.vec)
        while not (endInd == start):
            if endInd == len(self.vec):
                endInd = 0
                if start == 0:
                    break
            j = endInd
            while self.vec[j] in copy:
                j += 1
                if j == len(self.vec):
                    j = 0
            copy[endInd] = self.vec[j]
            endInd += 1
        self.vec = copy
        return self


class animal(uniqueIndiv):
    def __init__(self, size, maxEl=None, minEl=0):
        if not maxEl:
            maxEl = size - 1
        if maxEl - minEl < size:
            maxEl = minEl + size - 1
        self.vec = [i for i in range(size)]
        self.max = maxEl
        self.min = minEl
        self.brokenLeg = 0


def createFirtPopulation(
    indiv,
    popSize,
    individKwargs=None,
    greedyKwargs=None,
    greedy=None,
    greedyNum=None,
    check=None,
):
    individKwargs = individKwargs if individKwargs else {}
    greedyKwargs = greedyKwargs if greedyKwargs else {}
    numIndiv = popSize
    if greedy:
        if greedyNum:
            numIndiv -= greedyNum
        else:
            numIndiv //= 2
            greedyNum = popSize - numIndiv
    ans = []
    if check:
        for _ in range(numIndiv):
            toAdd = indiv(**individKwargs).fillRand()
            while not check(toAdd):
                toAdd.fillRand()
            ans.append(toAdd)
    else:
        ans = [indiv(**individKwargs).fillRand() for _ in range(numIndiv)]
    if greedy:
        for _ in range(greedyNum):
            ans.append(greedy(**greedyKwargs))
    return ans


# creating parents


def getParantsFight(
    population, adapt, paireNum=None, fightNum=None, dontRepeate=False, copy=True
):
    if not paireNum:
        paireNum = len(population) // 2
    if not fightNum:
        fightNum = paireNum // 4
        if fightNum == 0:
            fightNum = 1
    ans = []
    if dontRepeate:
        indList = [i for i in range(len(population))]
        for _ in range(paireNum):
            fight = [random.choice(indList) for i in range(fightNum)]
            indMom = max(fight, key=lambda a: adapt(population[a]))
            indList.remove(indMom)
            fight = [random.choice(indList) for i in range(fightNum)]
            indDed = max(fight, key=lambda a: adapt(population[a]))
            indList.remove(indDed)
            if copy:
                ans.append((deepcopy(population[indMom]), deepcopy(population[indDed])))
            else:
                ans.append((population[indMom], population[indDed]))
        return ans
    for _ in range(paireNum):
        fight = [random.choice(population) for i in range(fightNum)]
        Mom = max(fight, key=lambda a: adapt(a))
        fight = [random.choice(population) for i in range(fightNum)]
        Ded = max(fight, key=lambda a: adapt(a))
        if copy:
            ans.append((deepcopy(Mom), deepcopy(Ded)))
        else:
            ans.append((Mom, Ded))
    return ans


# crossover


def crossoverInsert(mom, ded, start=None, end=None):
    kid = deepcopy(ded)
    if not start:
        if end:
            start = 0
        else:
            start = len(kid.vec) // 2
    if not end:
        end = len(kid.vec)
    if start > end:
        start, end = end, start
    kid.insert(mom.vec[start:end], start, end)
    return kid

def crossoverReflect(mom, ded, start=None, end=None):
    kid = deepcopy(ded)
    if not start:
        if end:
            start = 0
        else:
            start = len(kid.vec) // 2
    if not end:
        end = len(kid.vec)
    if start > end:
        start, end = end, start

    reflect = {}
    for i in range(start, end):
        reflect[mom.vec[i]] = ded.vec[i]
        kid.vec[i] = mom.vec[i]
    for i in range(0, start):
        while kid.vec[i] in reflect:
            kid.vec[i] = reflect[kid.vec[i]]
    for i in range(end, len(kid.vec)):
        while kid.vec[i] in reflect:
            kid.vec[i] = reflect[kid.vec[i]]
    return kid

# mutation


def mutationSaltation(kid):
    firstInd, secondInd = (
        random.randint(0, len(kid.vec) - 1),
        random.randint(0, len(kid.vec) - 1),
    )
    while secondInd == firstInd:
        secondInd = random.randint(0, len(kid.vec) - 1)
    kid.vec[firstInd], kid.vec[secondInd] = kid.vec[secondInd], kid.vec[firstInd]
    return kid


# Selection


def selectionSpinThatWheeel(
    population, adaptation, size=None, keepBest=False, oneOfEach=False
):
    sumAdapt = sum([adaptation(i) for i in population])
    if not size:
        size = len(population) // 2
    ans = []
    if keepBest:
        ans.append(max(population, key=lambda a: adaptation(a)))
        size -= 1
    if oneOfEach:
        indList = [i for i in range(len(population))]
        if keepBest:
            indList.remove(max(indList, key=lambda a: adaptation(population[a])))
        for _ in range(size):
            ind = random.choices(
                indList,
                weights=[adaptation(population[i]) / sumAdapt for i in indList],
                k=1,
            )[0]
            ans.append(population[ind])
            indList.remove(ind)
        return ans
    ans += random.choices(
        population, weights=[adaptation(i) / sumAdapt for i in population], k=size
    )
    return ans


def selectionChase(rabbits, adaptation, wolfs, sumLeght, show=True):
    aliveRabbits = []
    aliveWolf = []
    rabbitsOrder = random.sample(rabbits, k=len(rabbits))
    wolfsOrder = random.sample(wolfs, k=len(wolfs))
    # breakpoint()
    for i in range(len(rabbitsOrder)):
        print(i)
        if i == len(wolfs):
            break
        wolf = wolfsOrder[i]
        rabbit = rabbitsOrder[i]
        wolfAdapt = adaptation(wolf)
        rabbitAdapt = adaptation(rabbit)
        if show:
            print(f"\t\t\twolf {wolf} = {100 / adaptation(wolf)}\n")
            print(f"\t\t\trabbit {rabbit} = {100 / adaptation(rabbit)}\n")
        if wolfAdapt > rabbitAdapt:
            aliveWolf.append(wolf)
            if show:
                print("\t\t\trabbit is dead&\n")
            continue
        start = 0
        while wolf.vec[start] != rabbit.vec[0]:
            start += 1
        wolf.vec = wolf.vec[start:] + wolf.vec[0:start]
        rabbitLive = True
        for i in range(1, len(rabbit.vec)):
            if sumLeght(wolf.vec[:i]) == sumLeght(rabbit.vec[:i]):
                aliveWolf.append(wolf)
                rabbitLive = False
                if show:
                    print("\t\t\trabbit is dead!\n")
                break
        if rabbitLive:
            aliveRabbits.append(rabbit)
            if show:
                print("\t\t\twolf is dead\n")
        if len(wolfs) > len(rabbits):
            aliveWolf += wolfsOrder[len(rabbits) :]
    return (aliveRabbits, aliveWolf)


def selectionChaseRand(rabbits, adaptation, wolfs, sumLeght, stuck, hDiff, lDiff, show=True):
    # breakpoint()
    rabbitsOrder = random.sample(rabbits, k=len(rabbits))
    wolfsOrder = random.sample(wolfs, k=len(wolfs))
    maxRabbit = max([adaptation(i) for i in rabbits])
    maxWolf = max([adaptation(i) for i in wolfs])
    # swithcRatio = 1.3
    if stuck and maxRabbit > maxWolf:
        bestRabbits = random.choices(
            rabbits, weights=[adaptation(i) for i in rabbits], k=len(rabbits) // 10 + 3
        )
        wolfsOrder.sort(key=lambda a: adaptation(a))
        wolfsOrder = wolfsOrder[: -(len(rabbits) // 10 + 3)]
        wolfsOrder += bestRabbits
        wolfsOrder = random.sample(wolfsOrder, k=len(wolfs))
    elif stuck and maxWolf > maxRabbit:
        bestWolfs = random.choices(
            wolfs, weights=[adaptation(i) for i in wolfs], k=len(rabbits) // 10 + 3
        )
        rabbitsOrder.sort(key=lambda a: adaptation(a))
        rabbitsOrder = rabbitsOrder[: -(len(rabbits) // 10 + 3)]
        rabbitsOrder += bestWolfs
        rabbitsOrder = random.sample(rabbitsOrder, k=len(rabbits))
    for i in range(min(len(rabbits), len(wolfs))):
        wolf = wolfsOrder[i]
        rabbit = rabbitsOrder[i]
        wolfAdapt = adaptation(wolf)
        rabbitAdapt = adaptation(rabbit)
        if show:
            print(f"\t\t\twolf {wolf} = {100 / adaptation(wolf)}\n")
            print(f"\t\t\trabbit {rabbit} = {100 / adaptation(rabbit)}\n")
        if rabbitAdapt / wolfAdapt < random.random():
            rabbitsOrder[i].brokenLeg += (min(
                [sumLeght(rabbit.vec[i : i + 2]) for i in range(len(rabbit.vec))]
            ) + max(
                [sumLeght(rabbit.vec[i : i + 2]) for i in range(len(rabbit.vec))]
            )) * (wolfAdapt / rabbitAdapt)
            if show:
                print("\t\t\trabbit is dead&\n")
            continue
        start = 0
        while wolf.vec[start] != rabbit.vec[0]:
            start += 1
        wolf.vec = wolf.vec[start:] + wolf.vec[0:start]
        rabbitLive = True
        incounter = 0
        for i in range(2, len(rabbit.vec)):
            rabbitWay = sumLeght(rabbit.vec[:i])
            wolfWay = sumLeght(wolf.vec[:i])
            if (
                wolf.vec[i] == rabbit.vec[i]
                and rabbitWay / wolfWay < hDiff
                and rabbitWay / wolfWay > lDiff
            ):
                deadChance = random.random()
                if deadChance + incounter > rabbitAdapt / wolfAdapt:
                    rabbitsOrder[i].brokenLeg += (min(
                        [
                            sumLeght(rabbit.vec[i : i + 2])
                            for i in range(len(rabbit.vec))
                        ]
                    ) + max(
                        [
                            sumLeght(rabbit.vec[i : i + 2])
                            for i in range(len(rabbit.vec))
                        ]
                    )) * (wolfAdapt / rabbitAdapt)
                    rabbitLive = False
                    if show:
                        print("\t\t\trabbit is dead!\n")
                    break
                incounter += 0.1
        if rabbitLive:
            wolfsOrder[i].brokenLeg += (min(
                [sumLeght(wolf.vec[i : i + 2]) for i in range(len(wolf.vec))]
            ) + max(
                [sumLeght(wolf.vec[i : i + 2]) for i in range(len(wolf.vec))]
            ))* (rabbitAdapt / wolfAdapt)
            if show:
                print("\t\t\twolf is dead\n")
    return (rabbitsOrder, wolfsOrder)


def selectionEmpty(population, adapt):
    return population


class GeneticCommivAlg:
    def __init__(
        self,
        name,
        size,
        populationSize,
        mutationRateGen,
        createFirtPopulation=createFirtPopulation,
        indiv=animal,
        individKwargs=None,
        greedyKwargs=None,
        greedy=None,
        greedyNum=None,
        check=None,
    ):
        self.indiv = indiv
        self.name = name
        self.size = size
        self.populationSize = populationSize
        #breakpoint()
        self.population = createFirtPopulation(
            indiv, populationSize, individKwargs, greedyKwargs, greedy, greedyNum, check
        )
        self.step = 0
        self.mutate = mutationRateGen()
        self.results = []

    def evolve(
        self,
        adaptation,
        getParents,
        crossover,
        mutation,
        selection,
        getParentsKwargs={},
        selectionKwargs={},
    ):
        mutationRate = self.mutate.__next__()
        #breakpoint()
        best = max(self.population, key=lambda a: adaptation(a))
        self.results.append(100 / adaptation(best))
        for parents in getParents(self.population, **getParentsKwargs):
            kid = crossover(parents[0], parents[1])
            mutateChance = random.random()
            if mutateChance < mutationRate:
                mutation(kid)
            self.population.append(kid)
            kid = crossover(parents[1], parents[0])
            mutateChance = random.random()
            if mutateChance < mutationRate:
                mutation(kid)
            self.population.append(kid)
        self.population = selection(self.population, adaptation, **selectionKwargs)
        self.step += 1
        return self.population


def runChase(
    rabbits,
    wolfs,
    selection,
    adaptationSum,
    steps,
    populationSize,
    sumPart,
    evolveKw,
    selectionKwargs,
    lDiff,
    hDiff,
    draw=False,
):
    stuckNum = populationSize / 10
    stuckStep = 0
    bestRabbit = max(rabbits.population, key=lambda a: adaptationSum(a))
    bestWolf = max(wolfs.population, key=lambda a: adaptationSum(a))
    best = max(bestRabbit, bestWolf, key=lambda a: adaptationSum(a))
    while rabbits.step < steps:
        wolfs.evolve(**evolveKw)
        rabbits.evolve(**evolveKw)
        stuck = stuckStep > stuckNum
        if stuck:
            stuckStep = 0
        newPopulation = selectionChaseRand(
            rabbits.population, adaptationSum, wolfs.population, sumPart, stuck, hDiff, lDiff, False
        )
        rabbits.population = newPopulation[0]
        wolfs.population = newPopulation[1]
        rabbits.population = selection(rabbits.population, **selectionKwargs)
        wolfs.population = selection(wolfs.population, **selectionKwargs)
        newWolf = max(wolfs.population, key=lambda a: adaptationSum(a))
        newRabbit = max(rabbits.population, key=lambda a: adaptationSum(a))
        if adaptationSum(bestRabbit) == adaptationSum(newRabbit) and adaptationSum(
            bestWolf
        ) == adaptationSum(newWolf):
            stuckStep += 1
        else:
            stuckStep = 0
        bestWolf = newWolf
        bestRabbit = newRabbit
        best = max(bestRabbit, bestWolf, best, key=lambda a: adaptationSum(a))
    if draw:
        return [{rabbits.name: rabbits.results, wolfs.name: wolfs.results}, best]
    else:
        return (best, adaptationSum(best))
