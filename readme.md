# Installation and running

1. Install Java 8 SDK and add JAVA_PATH
2. Install Maven and add its bin folder to PATH
3. In project folder run `mvn install`
4. Go to folder `./targer` and run program: `java -cp NeuralNetwork-1.0.jar Main`

Run command example:
	`java -cp NeuralNetwork-1.0.jar Main -i 64 -o 1 --training-dataset "/Path/To/TrainingDataset.tra" --testing-dataset "/Path/To/TestingDataset.tes" -l 20 -bs 10 -le 0.02 -me 5000 --randomize --normalize --momentum -tge 0.05 -sm 0.05`

####Available arguments:
```
	(required) 	-i <n> 						- number of inputs
	(required) 	-o <n> 						- number of ouputs

	(required)	--training-dataset <path> 	- file with training dataset
	(required)	--testing-dataset <path> 	- file with testing dataset
				--stochastic				- use stochastic gradient descent (default: mini-batch)
				-bs <n>						- size of batch of mini-batch gradient descent (default: 10)
				--randomize					- enable input batch randomization
				--normalization				- enable input batch normalization

				-l <"n, n, n">				- hidden layers of network
                                (example: "20, 10, 10", for network with 3 hidden layers,
                                with 20, 10 and 10 neurons)
				-af [sigmoid, tanh]			- set activation function (default: sigmoid)
				-le <n>						- set learning rate	(default: 0.01)
				--momentun					- enable momentum optimization
				--dropout					- enable dropout optimization
				-me <n>						- set maximum of epochs (default: 1000)
				-tge <n>					- target global error (default: 0.1)
				-sm <n>						- success mistake, using in comparison of
                            network output and expected output (default: 0.1)
```
