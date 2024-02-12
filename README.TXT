Make sure you have deap, numpy, and matplotlib installed. They can be installed with pip.

Edit the ARGS file to change the arguements (ex. population size) of the GP.

Edit the INPUT_VALUES file to change the x-values and y-values of the GP.
To quickly randomly generate values, run the list_gen.py file.

	python list_gen.py

It will erase the old INPUT_VALUES file and create a brand new one with different values.

Run the GP with

	python symreg.py

It will output a table into standard out, a plot in the file plot.png, a file HOF 
containing the Hall Of Famers who are the best individuals ever in the GP.
