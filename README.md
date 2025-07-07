# Evolutionary Multi Agent MLP Training

This repository contains code for training multi-layer perceptron (MLP) agents using evolutionary strategies. Each agent has a three pixel input, has a small memory and outputs a rotation and a movement command. The agents are trained to avoid each other and to move towards food.

Since crossover operations for MLPs are not well defined, each time an agent dies, a new agent is clone from the top 10% of the population. The new agent is mutated by randomly changing the weights of the MLP.

Suprisingly, this approach works well and after at the mark of the 1000th agent the agents are able to avoid each other and move towards food.


## Running the Code

To run the code, you need to rust installed. You can install it from [rustup.rs](https://rustup.rs/).

After installing Rust, you can run the code with the following command:

```bash
cargo run
```
