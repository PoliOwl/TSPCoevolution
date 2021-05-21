import EvGenLab as eg
import random
import pandas as pd
from matplotlib import pyplot as plt


WAYS = []
MUTATION_RATE = 0.02
MUTATION_STEP = 0.01


def adaptationSum(indiv):
    ans = 0
    for i in range(len(indiv.vec) - 1):
        ans += WAYS[indiv.vec[i]][indiv.vec[i + 1]]
    ans += WAYS[indiv.vec[-1]][indiv.vec[0]]
    return 100 / ans


def adaptationBrok(indiv):
    ans = 0
    for i in range(len(indiv.vec) - 1):
        ans += WAYS[indiv.vec[i]][indiv.vec[i + 1]]
    ans += WAYS[indiv.vec[-1]][indiv.vec[0]] + indiv.brokenLeg
    return 100 / ans


def sumPart(vec):
    ans = 0
    for i in range(len(vec) - 1):
        ans += WAYS[vec[i]][vec[i + 1]]
    return ans


@eg.static_variables(mutRate=MUTATION_RATE, mutStep=MUTATION_STEP)
def mutateRate():
    global MUTATION_STEP
    # mutateRate.mutRate = mutationRate
    # mutateRate.mutStep = mutationStep
    while True:
        if mutateRate.mutRate > 0.9:
            mutateRate.mutRate = 0.9
            MUTATION_STEP = -0.01
        elif mutateRate.mutRate < 0.05:
            mutateRate.mutRate = 0.05
            MUTATION_STEP = 0.05
        yield mutateRate.mutRate
        mutateRate.mutRate += mutateRate.mutStep


def runChase(
    size,
    populationSize,
    mutationRate,
    crossover,
    getParents,
    getParentsKw,
    mutation,
    selection,
    steps,
    hDiff,
    lDiff,
    draw=False,
    optimum=None,
    greed=None,
    adaptationBrok=adaptationBrok,
    adaptationSum=adaptationSum,
    sumPart=sumPart,
):
    MUTATION_RATE = mutationRate
    wolfs = eg.GeneticCommivAlg(
        "wolf",
        size,
        populationSize,
        individKwargs={"size": size},
        mutationRateGen=mutateRate,
    )
    rabbits = eg.GeneticCommivAlg(
        "rabbits",
        size,
        populationSize,
        individKwargs={"size": len(WAYS)},
        mutationRateGen=mutateRate,
    )
    evolveKw = {
        "adaptation": adaptationBrok,
        "getParents": getParents,
        "crossover": crossover,
        "mutation": mutation,
        "selection": eg.selectionEmpty,
        "getParentsKwargs": getParentsKw,
        "selectionKwargs": {},
    }
    selectionKw = {"adaptation": adaptationBrok, "size": populationSize}
    ans = eg.runChase(
        rabbits,
        wolfs,
        selection,
        adaptationSum,
        steps,
        populationSize,
        sumPart,
        evolveKw,
        selectionKw,
        hDiff,
        lDiff,
        draw,
    )
    if draw:
        df = pd.DataFrame(ans[0])
        df["optimum"] = optimum
        df["greedy"] = greed
        ax = df.plot()
        ax.set_xlabel("evolution steps")
        ax.set_ylabel("adaptation")
        plt.show()
        return ans[1]
    return ans


if __name__ == "__main__":
    with open("for_com.txt") as file:
        WAYS = [[float(i) for i in line.strip().split()] for line in file]
    getParents = eg.getParantsFight
    getParentsKw = {}
    getParentsKw["adapt"] = adaptationBrok
    mutation = eg.mutationSaltation
    crossover = eg.crossoverInsert
    populationSize = 500
    ans = runChase(
        size=len(WAYS[0]),
        populationSize=populationSize,
        mutationRate=0.2,
        crossover=crossover,
        getParents=getParents,
        getParentsKw=getParentsKw,
        mutation=mutation,
        selection=eg.selectionSpinThatWheeel,
        steps=300,
        hDiff=1.5,
        lDiff=0.5,
        draw=True,
        optimum=699,
        greed=864,
    )
    print(f"answer is {ans} = {100 / adaptationSum(ans)}")
