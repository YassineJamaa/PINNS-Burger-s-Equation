
# Physics-Informed Neural Network for Solving the Burgers' Equation

## Project Overview
This project explores the application of **Physics-Informed Neural Networks (PINNs)** to solve the **Burgers' equation**, a fundamental partial differential equation that arises in various areas of fluid dynamics and nonlinear acoustics. The model is implemented using PyTorch and utilizes a combination of two optimizers—**Adam** and **L-BFGS**—to achieve accurate results.

## Problem Statement
The Burgers' equation is defined as:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}
$$

where:
- $\( u(x, t) \)$ represents the solution of the PDE,
- \( \nu \) is the viscosity parameter.

In this project, the equation is solved using a **Physics-Informed Neural Network** by incorporating the equation's constraints into the loss function. The model takes as input the spatial and temporal coordinates \((x, t)\) and outputs the solution \( u(x, t) \).

## Loss Function and Backpropagation
The custom loss function used in this project is designed to satisfy both the initial and boundary conditions as well as the governing physics of the Burgers' equation. The **data loss** ensures that the predicted solution adheres to the initial and boundary conditions of the PDE. This is computed as the mean squared error (MSE) between the network's prediction and the known values at \( t=0 \) (initial condition) and at the boundaries \( x=-1 \) and \( x=1 \).

The **physics loss** is derived from the residual of the Burgers' equation itself. During backpropagation, the network calculates the first and second derivatives of its predictions with respect to the input coordinates \((x, t)\) using **automatic differentiation**. The loss is then computed as the MSE between the left-hand side (LHS) and right-hand side (RHS) of the equation:

\[
\text{loss\_pde} = \text{MSE} \left( \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x}, \, \nu \frac{\partial^2 u}{\partial x^2} \right)
\]

The overall loss function is defined as the sum of the **initial condition loss** and the **PDE loss**:

\[
\text{loss} = \text{loss\_initial\_condition} + \text{loss\_pde}
\]

This ensures that the network not only fits the data but also respects the physical law encoded in the PDE. The initial condition loss is calculated as the MSE between the model's prediction and the initial state \( u(x, 0) \). The total loss guides the network to learn solutions that are physically consistent and satisfy the dynamics of the Burgers' equation.

## Project Structure
- `NN` Class: A custom Multi-Layer Perceptron (MLP) architecture is designed to approximate the solution of the Burgers' equation.
- **Custom Loss Function**: Combines the data loss (based on initial and boundary conditions) and the physics loss (based on the PDE's governing equation) to guide the network towards a physically consistent solution.
- **Training Strategy**:
  - **Stage 1**: Training with the Adam optimizer for rapid convergence.
  - **Stage 2**: Fine-tuning with the L-BFGS optimizer for precise results and to minimize the physics residual.

## References
This work is inspired by the research conducted by **Raissi et al.** on Physics-Informed Neural Networks, which provides a foundation for solving nonlinear partial differential equations using deep learning techniques:

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2017). *Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations*. arXiv preprint [arXiv:1711.10561](https://arxiv.org/abs/1711.10561).
