# Genetic Algorithm Implementation Guide

This document provides detailed information about the Genetic Algorithm (GA) implementation in IDPR.

## Overview

The GA implementation solves the drone positioning and routing problem through an evolutionary approach. The algorithm maintains a population of candidate solutions (drone topologies) and evolves them over generations using selection, crossover, and mutation operators.

## Algorithm Structure

The GA is implemented as a pipeline of seven modular steps (Passos):

### Passo 1: Population Initialization

**Purpose**: Generate the initial population of random drone topologies.

**Key Features:**
- Creates random drone positions on the grid
- Ensures no two drones occupy the same position
- Generates random communication links between drones
- Validates basic connectivity constraints

**Parameters:**
- Population size (P): typically 300-500
- Grid size (G): typically 100×100
- Initial drone count (N): typically 5-30

**Output**: Initial population of P individuals, each with N drones and random topology

### Passo 3: Fitness Evaluation

**Purpose**: Calculate the fitness score for each individual in the population.

**Fitness Function** (Lexicographic Multi-Objective):

The fitness is computed using a hierarchical objective function:

1. **Primary Objective**: Maximize reliability
   - End-to-end packet delivery probability ≥ 99.999%
   - Computed using Friis propagation model with log-normal fading

2. **Secondary Objective**: Minimize maximum latency
   - Latency = (number of hops) × (decoding time per hop)
   - Penalizes long paths

3. **Tertiary Objective**: Minimize number of drones
   - Reduces deployment cost and complexity

**Fitness Score**:
$$f(i) = \frac{1}{w_1 \cdot R_{\text{penalty}} + w_2 \cdot L_{\text{max}} + w_3 \cdot N_{\text{drones}}}$$

Where:
- $R_{\text{penalty}}$ = 0 if reliability ≥ threshold, else large penalty
- $L_{\text{max}}$ = maximum end-to-end latency
- $N_{\text{drones}}$ = number of drones in topology
- $w_1, w_2, w_3$ = weights (typically 1.0, 0.5, 0.3)

### Passo 4: Selection

**Purpose**: Select the best individuals to become parents for reproduction.

**Selection Methods:**

1. **Tournament Selection** (recommended):
   - Select k random individuals from population
   - Choose the one with highest fitness
   - Repeat to select two parents
   - Parameter: tournament size k (typically 3-5)

2. **Roulette Wheel Selection**:
   - Probability of selection proportional to fitness
   - Better individuals more likely to be selected
   - Can lead to premature convergence

**Elitism**:
- Preserve the top E individuals (typically 10-20% of population)
- Ensures best solutions are not lost

### Passo 5: Crossover

**Purpose**: Create offspring by combining genetic material from two parents.

**Crossover Operators:**

1. **Uniform Crossover**:
   - For each drone position: randomly choose from parent 1 or parent 2
   - Probability: 50% from each parent
   - Good for maintaining diversity

2. **Single-Point Crossover**:
   - Select random crossover point
   - First part from parent 1, second part from parent 2
   - Preserves building blocks

3. **Two-Point Crossover**:
   - Select two crossover points
   - Alternate between parents
   - Better preserves longer sequences

**Crossover Rate**: Typically 0.8 (80% of offspring created via crossover, 20% copied from parents)

**Repair Mechanism**:
- After crossover, validate offspring:
  - Remove duplicate drone positions
  - Ensure connectivity constraints
  - Repair invalid topologies

### Passo 6: Mutation

**Purpose**: Introduce random variations to maintain genetic diversity.

**Mutation Operators:**

1. **Drone Position Mutation**:
   - Select random drone
   - Move to random position on grid
   - Probability: 0.1-0.2 per drone

2. **Link Mutation**:
   - Add random link between two drones
   - Remove random existing link
   - Probability: 0.05-0.15 per link

3. **Swap Mutation**:
   - Select two random drones
   - Swap their positions
   - Probability: 0.05-0.1

**Mutation Rate**: Typically 0.1-0.2 (10-20% of offspring mutated)

**Adaptive Mutation**:
- Increase mutation rate if population converges prematurely
- Decrease mutation rate as algorithm progresses
- Helps escape local optima

### Passo 7: Replacement & Elitism

**Purpose**: Form the new generation for the next iteration.

**Replacement Strategies:**

1. **Generational Replacement**:
   - Entire population replaced by offspring
   - Preserves best individuals via elitism
   - Good for exploration

2. **Steady-State Replacement**:
   - Replace only worst individuals
   - Slower convergence but maintains diversity
   - Better for fine-tuning

**Elitism Implementation**:
- Copy top E individuals directly to next generation
- Remaining (P-E) individuals are offspring
- Ensures best fitness never decreases

## Compilation and Execution

### Prerequisites

- C++11 or later compiler (g++, clang, or MSVC)
- Standard library support
- No external dependencies

### Compilation

**Basic compilation:**
```bash
cd src/ga
g++ -o ga_optimizer Passo*.cpp -std=c++11 -O3
```

**With debugging symbols:**
```bash
g++ -o ga_optimizer Passo*.cpp -std=c++11 -g -O2
```

**With optimization flags:**
```bash
g++ -o ga_optimizer Passo*.cpp -std=c++11 -O3 -march=native -flto
```

### Execution

**Basic execution:**
```bash
./ga_optimizer <population_size> <grid_size>
```

**Examples:**
```bash
# Small problem (100 drones, 50×50 grid)
./ga_optimizer 100 50

# Medium problem (300 drones, 100×100 grid)
./ga_optimizer 300 100

# Large problem (500 drones, 200×200 grid)
./ga_optimizer 500 200
```

### Output

The GA generates output files for each generation:

- **I__1, I__2, ..., I__N**: Individual topology files
- **fitness.log**: Fitness values per generation
- **best_solution.txt**: Best solution found

## Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Population Size | 300 | 50-1000 | Number of individuals per generation |
| Generations | 200 | 10-1000 | Number of generations to evolve |
| Crossover Rate | 0.8 | 0.5-0.95 | Probability of crossover vs copy |
| Mutation Rate | 0.15 | 0.05-0.5 | Probability of mutation per individual |
| Tournament Size | 3 | 2-10 | Number of individuals in tournament |
| Elitism Ratio | 0.1 | 0.05-0.2 | Fraction of best individuals preserved |
| Grid Size | 100 | 50-500 | Size of operational grid |
| Drone Count | 25 | 5-100 | Number of drones to deploy |

## Performance Tuning

### For Better Solutions (Slower Convergence)

```bash
# Increase population and generations
./ga_optimizer 500 100  # Larger population
# Modify source: increase GENERATIONS to 500
# Decrease mutation rate: MUTATION_RATE = 0.05
```

### For Faster Convergence (Potentially Worse Solutions)

```bash
# Decrease population and generations
./ga_optimizer 100 100  # Smaller population
# Modify source: decrease GENERATIONS to 50
# Increase mutation rate: MUTATION_RATE = 0.3
```

### For Balanced Performance

```bash
# Recommended settings
./ga_optimizer 300 100
# GENERATIONS = 200
# MUTATION_RATE = 0.15
# CROSSOVER_RATE = 0.8
```

## Monitoring Progress

### Fitness Convergence

The algorithm should show:
1. Rapid improvement in first 20-30% of generations
2. Gradual improvement in middle 40%
3. Plateau in final 30% (convergence)

If fitness plateaus too early, increase mutation rate or population size.

### Population Diversity

Monitor the diversity of solutions:
- High diversity: population exploring different regions
- Low diversity: population converging to local optimum

If diversity is too low, increase mutation rate or use adaptive mutation.

## Troubleshooting

### Issue: Compilation Errors

**Problem**: "error: 'Individuo' was not declared in this scope"

**Solution**: Ensure all header files are present and included correctly. Verify that Passo*.h files are in the same directory.

### Issue: Segmentation Fault

**Problem**: Program crashes during execution

**Solution**: 
- Check grid size and population size are reasonable
- Verify sufficient memory available
- Compile with debugging symbols and use gdb

### Issue: Poor Solutions

**Problem**: Final solution has low fitness

**Solution**:
- Increase population size
- Increase number of generations
- Decrease mutation rate (for fine-tuning)
- Verify fitness function is correct

### Issue: Slow Execution

**Problem**: Program runs too slowly

**Solution**:
- Compile with optimization flags (-O3)
- Reduce population size or generations
- Check for algorithmic bottlenecks with profiler

## Comparison with RL

| Aspect | GA | RL |
|--------|----|----|
| Training Time | 1-10 hours | 40-96 hours |
| Solution Quality | Often better | Good, generalizable |
| Inference Speed | Slow (re-optimize) | Fast (forward pass) |
| Generalization | No | Yes |
| Implementation | C++ (fast) | Python (flexible) |

## Advanced Topics

### Parallel GA

The current implementation is sequential. For parallel execution:
1. Distribute population across multiple processes
2. Evaluate fitness in parallel
3. Synchronize for selection and replacement

### Adaptive Parameters

Implement parameter adaptation:
- Increase mutation rate if fitness plateaus
- Decrease crossover rate if diversity is low
- Adjust tournament size based on convergence

### Multi-Objective Optimization

Extend to Pareto-optimal solutions:
- Maintain population of non-dominated solutions
- Use NSGA-II or similar algorithm
- Trade off between multiple objectives

## References

- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization and Machine Learning.
- Mitchell, M. (1998). An Introduction to Genetic Algorithms.
- Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms.
