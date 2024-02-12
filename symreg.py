import operator
import math
import random
from functools import partial

import numpy
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools, gp

with open("ARGS", "r") as file:
	file_list = [f for f in file.readlines() if not f.startswith('#')]
	file_list = [f.strip() for f in file_list]

	POP_SIZE = int(file_list[0])
	P_CROSSOVER = float(file_list[1])
	P_MUTATION = float(file_list[2])
	MAX_GENERATIONS = int(file_list[3])
	HOF_SIZE = int(file_list[4])
	T_SIZE = int(file_list[5])
	DEPTH = int(file_list[6])
	SEED = int(file_list[7])


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

# 1 for a 1-dimensional symbolic regression problem.
pset = gp.PrimitiveSet("MAIN", 1)

# Adding nodes for the tree.
# Number represents arity of operator.
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

# Adding terminal nodes that are determined at runtime as either a -1, 0, or 1.
pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
pset.renameArguments(ARG0='x')

# Negative weight represents a minimization problem.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Individuals are represented as trees.
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# We create individuals as expressions first.
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=DEPTH)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

X_VALUES=[]
Y_VALUES=[]

with open("INPUT_VALUES", "r") as file:
	file_list = [f for f in file.readlines() if not f.startswith('#')]
	file_list = [f.strip() for f in file_list]
	X_VALUES=file_list[0].split(",")
	X_VALUES=[float(x) for x in X_VALUES]
	Y_VALUES=file_list[1].split(",")
	Y_VALUES=[float(y) for y in Y_VALUES]

LENGTH=len(X_VALUES)

def evalSymbReg(individual, points):
    func = toolbox.compile(expr=individual)
    sqerrors = [0 for i in range(LENGTH)]
    #sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
	
    for i in range(LENGTH):
        sqerrors[i] = (func(points[i]) - Y_VALUES[i])**2

    return math.fsum(sqerrors),


toolbox.register("evaluate", evalSymbReg, points=X_VALUES)
toolbox.register("select", tools.selTournament, tournsize=T_SIZE)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=17)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

random.seed(SEED)

def main():

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(HOF_SIZE)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=mstats, halloffame=hof, verbose=True,)
    bestVals, avgVals = log.chapters["fitness"].select("min", "avg")
    
    plt.plot(bestVals, color="red")
    plt.plot(avgVals, color="green")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Best and Average Fitnesses over Generations")
    plt.savefig("plot.png")

    with open("HOF", "w") as file:
        for h in hof:
    	    file.write(f"{str(h)}\n")
	
    return pop, log, hof

if __name__ == "__main__":
    main()
