import matplotlib.pyplot as plt


    plt.plot(bestVals, color="red")
    plt.plot(avgVals, color="green")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Best and Average Fitnesses over Generations")
    plt.savefig("plot.png")
