import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt

n_dim=1
n_params=1
n_training_points=1000

def integrand(p_i,x_i): #Returns actual integral value
    return p_i

def numerical_calc(params, x): #Generates tensor of actual integral values
    return torch.stack([
        integrand(p_i, x_i).reshape(())
        for p_i, x_i in zip(params, x)
    ]).unsqueeze(1)

def preprocess_integrand(params,x,y):

    #Normalisation
    x_mid = torch.full_like(x, 0.5)
    f_mid = numerical_calc(params, x_mid)
    y_normalised = y / f_mid

    #Korobov transform to come

    return y_normalised, f_mid

def gen_training_points():

    params_test=torch.rand(n_training_points,n_params)*10
    x_test= torch.rand(n_training_points,n_dim)
    y_test=numerical_calc(params_test,x_test)

    return params_test, x_test, y_test

def run_nn_training(model, n_iterations=5000, learning_rate=0.001):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )
    losses = []

    for iteration in range(n_iterations):

        params_test, x_test, y_raw = gen_training_points()
        y_test,f_mid = preprocess_integrand(params_test,x_test,y=y_raw)
        x = x_test.clone().detach().requires_grad_(True)
        inputs = torch.cat([params_test, x], dim=n_dim)
        output = model(inputs) # Forward pass

        dN_dx = torch.autograd.grad( #Calculates derivate w.r.t x - ONLY WORKS FOR 1D INTEGRALS
            outputs=output,
            inputs=x,
            grad_outputs=torch.ones_like(output),
            create_graph=True
        )[0]

        loss = nn.functional.mse_loss(dN_dx, y_test)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if iteration % 500 == 0:
            print(
                f"Iteration {iteration}: "
                f"loss = {loss.item():.8f}"
            )
    return losses

class PrimitiveNetwork(nn.Module):

    def __init__(
        self,
        n_parameters,
        n_dimensions,
        hidden_width=16,
        n_hidden_layers=2
    ):
        super().__init__()
        input_size = n_parameters + n_dimensions
        layers = []

        #First hidden layer
        layers.append(nn.Linear(input_size, hidden_width))
        layers.append(nn.Tanh())

        #Additional hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh())

        #Final layer
        layers.append(nn.Linear(hidden_width, 1))

        #Combine layers
        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)

model = PrimitiveNetwork(
    n_parameters=n_params,
    n_dimensions=n_dim,
    hidden_width=16,
    n_hidden_layers=2
)

losses = run_nn_training(
    model,
    n_iterations=5000,
    learning_rate=0.001
)

p_test = 3.5 #Temporary test param
p_tensor = torch.tensor([[p_test]])
x0 = torch.tensor([[0.0]])
x1 = torch.tensor([[1.0]])

with torch.no_grad():
    N0 = model(torch.cat([p_tensor, x0], dim=n_dim))
    N1 = model(torch.cat([p_tensor, x1], dim=n_dim))
    normalised_integral = (N1 - N0).item()

    #De-normalising integral
    f_mid_test = numerical_calc(
        p_tensor,
        torch.tensor([[0.5]])
    ).item()
    integral_prediction = normalised_integral * f_mid_test

print("Predicted integral:", integral_prediction)
print("Exact integral:", p_test)

#Plotting loss just to see
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Derivative MSE loss")
plt.yscale("log")
plt.show()