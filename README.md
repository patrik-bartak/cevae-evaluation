# Evaluation of Causal Machine Learning Methods
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is part of the B.Sc. thesis project of Patrik Barták for CSE3000 in Q4 2022. Some code is shared with the repository of Matej Havelka.
It accompanies the B.Sc. thesis freely available [here](https://repository.tudelft.nl/islandora/object/uuid%3A632eec99-2494-4ead-8455-d7ad5c1d18c9?collection=education), hosted in the TU Delft repository.

## How to run it
To run the experiment you can run the [main script](evaluator/main.py). 
Afterwards you should be able to find the results in newly generated directories, most importantly in the parameterization directory.

## How to extend it
To add a new model you need to extend the CausalMethod class in [the appropriate class](causal_effect_methods.py). Then add a new function to the [experiment builder](sample/experiment.py) that adds the causal method.
Afterwards you can construct the experiment with whatever data generators there are.

To add a new generator you need to create a new function in [experiment builder](sample/experiment.py) where you define the necessary functions to generate that data. With that you can add it to any experiment as you would with other generators.

## Authors
- Patrik Barták - P.Bartak@student.tudelft.nl
- Matej Havelka - M.Havelka@student.tudelft.nl
