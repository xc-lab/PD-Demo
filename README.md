# PD-Demo

<img src="https://github.com/xc-lab/PD-Demo/blob/main/demo-1.jpg?raw=true" width="800">

## Hyperparameter Settings
### Participant ID

Each participant is assigned a unique identifier that is used to load the corresponding handwritten signal data.

### Test Pattern

For every participant, handwriting signals are collected under two different test modes based on predefined drawing templates:

SST (Static Stimulus Test):
The template remains continuously visible and stationary throughout the drawing task.

DST (Dynamic Stimulus Test):
The template intermittently flashes, meaning it repeatedly appears and disappears during the drawing task.

These two modes are designed to capture potential variations in neuromotor control under different visual–motor conditions.

### Diagnostic Model: PointNet

The diagnostic classification is performed using a PointNet-based model, which processes the 3D handwriting point-cloud representation.

### Perturbation Sample Size

Given an input handwriting sample A, after PointNet produces its diagnostic prediction, a local perturbation–based explanation is performed.
To interpret the black-box behavior of PointNet around input A, a set of perturbed samples is generated within the local neighborhood of A. These perturbed samples are then used to train a simple, interpretable linear surrogate model that approximates PointNet's decision boundary near A.

The perturbation sample size parameter controls how many such locally generated samples are created for surrogate-model training.
