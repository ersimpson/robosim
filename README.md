# ECE 227 Final Project Spring 2024

## Description

This project simulates a simple 2D robot navigation system. The robot must navigate from a starting point to an ending point without being able to 
directly use the target coordinates. Instead, the robot has access to a overlaying graph of agents which can communicate information about the target coordinates to help guide the robot.

## Installation

To install this project, follow these steps:

1. Clone the repository.
2. Create a virtual environment `python3 -m venv venv && source venv/bin/activate`.
3. Install the dependencies `pip install -r requirements.txt`.

## Usage

### Running the Simulation (AStar Algorithm)

```
python3 robosim.py run -N 10
```

### Running the Simulation (Custom Model Algorithm v1)

```
python3 robosim.py run -N 10 -m model.pth -v v1
```

### Generating Data for Custom Model Training

```
python3 robosim.py gen-trajectories -N 1000
```

### Training Custom Model

```
python3 robosim.py train -n 100 -f ./trajectories.csv -o model.pth
```

## Authors

- Eric Simpson, ersimpson@ucsd.edu