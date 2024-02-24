# Pytorch and Code Modularization

In this exercise we have moved the code away from jupyter notebook into module files for better code readability and re-usability.


## model.py

Now the model file is moved to model.py as we could reuse the same network

## utils.py

This file encompasses all the utility functions used throughout our experiment, the functions are explained below

- ***get_device*** - sets the device to cuda if gpu is available else set it to cpu, we can also ecplicitly mention the device
- ***display_batch*** - display the images in a grid from the first batch of the given data loader
- ***load_dataset*** - load train or test datasets
- ***train*** - network train routine
- ***test*** - network test routine
- ***train_model*** - wrapper function to train and validate the network
- ***plot_metrics** - function to plot and compare the train and test metrics