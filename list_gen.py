import random

LIST_SIZE=12
LIST=[]

def fill_list(no_sort):
	for i in range(LIST_SIZE):
		LIST.append(random.random()*2 - 1)
	if not no_sort:
		LIST.sort()

def conv_list():
	for l in range(LIST_SIZE):
		if l > 0:
			LIST[l] = str(LIST[l])[:4]
		else:
			LIST[l] = str(LIST[l])[:5]

with open("INPUT_VALUES", "w") as file:
	file.write("# First row is independent x-values, second row is the corresponding y-values.\n")

	for row in range(2):
		fill_list(row)
		conv_list()
		file.write(",".join(LIST))
		file.write("\n")
		LIST=[]
