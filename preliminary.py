import torch
from torch import nn as nn
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


#Test parameter for integral prediction
p_tensor = torch.tensor([[3,1]],dtype=torch.float32)

#Setting parameters for training
n_dim=2
n_params=p_tensor.size(dim=1)
n_training_points=1000
n_iterations=5000

#Calculating actual integral value for test parameter
def integral_value(params, x):
    return 0.5*(params[0]*x[1]*x[0]**2+params[1]*x[0]*x[1]**2)

#Obtaining training data

def integrand(p_i,x_i): #Returns actual integral value
    return p_i[0]*x_i[0]+p_i[1]*x_i[1]

def numerical_calc(params, x): #Generates tensor of actual integral values
    return torch.stack([
        integrand(p_i, x_i).reshape(())
        for p_i, x_i in zip(params, x)
    ]).unsqueeze(1)

def gen_training_points():

    params_test = 0.1 + torch.rand(n_training_points, n_params) * 9.9
    t_test= torch.rand(n_training_points,n_dim)

    return params_test, t_test


#Preprocessing (Korobov transform and normalisation of integrand)

def korobov_transform(t):
    return t**2*(3-2*t)

def korobov_weight(t):
    return 6*t*(1-t)

def preprocess_integrand(params,t):

    #Korobov transform
    x = korobov_transform(t)
    jacobian=korobov_weight(t).prod(dim=1, keepdim=True)
    y_raw= numerical_calc(params, x)

    #Normalisation
    x_mid = torch.full_like(t, 0.5)
    f_mid = numerical_calc(params, x_mid)
    y_normalised = jacobian*y_raw/f_mid

    return y_normalised, f_mid


#Training loop

def n_dim_derivative(output,t):
    derivative = output

    for dimension in range(t.shape[1]):
        gradient = torch.autograd.grad(
            outputs=derivative,
            inputs=t,
            grad_outputs=torch.ones_like(derivative),
            create_graph=True
        )[0]
        derivative = gradient[:, dimension:dimension + 1]

    return derivative

def run_nn_training(model, learning_rate=0.001):

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    losses = []

    for iteration in range(n_iterations):

        params_test, t_test = gen_training_points()
        y_test,f_mid = preprocess_integrand(params_test,t_test)
        t = t_test.clone().detach().requires_grad_(True)

        inputs = torch.cat([params_test, t], dim=1)
        output = model(inputs) # Forward pass
        dN_dt=n_dim_derivative(output,t)

        loss = nn.functional.mse_loss(dN_dt, y_test)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if iteration % 500 == 0:
            print(f"Iteration {iteration}: loss = {loss.item():.8f}")
    return losses

class PrimitiveNetwork(nn.Module):

    def __init__(self,n_parameters,n_dimensions,hidden_width=16,n_hidden_layers=2):
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


#Post-processing

def post_processing (model, params):

    normalised_integral = torch.zeros((1,1),
    dtype=params.dtype,device=params.device)
        
    with torch.no_grad():
        for corner in product([0.0, 1.0],repeat=n_dim): #All corners of hypercube

            c = torch.tensor(corner,
                dtype=params.dtype,device=params.device).reshape(1, n_dim)
            
            inputs = torch.cat([params, c], dim=1)
            N_corner = model(inputs)
            sign = (-1)**(n_dim-sum(corner))

            normalised_integral+=sign*N_corner

    x_mid = torch.full((1,n_dim),0.5,
        dtype=params.dtype,device=params.device)
    f_mid = numerical_calc(params,x_mid)

    integral_prediction=normalised_integral*f_mid

    return integral_prediction.item()


model = PrimitiveNetwork(
    n_parameters=n_params,
    n_dimensions=n_dim,
    hidden_width=16,
    n_hidden_layers=2
)

losses = run_nn_training(model,learning_rate=0.001)

integral_prediction = post_processing(model, p_tensor)

print("Predicted integral:", integral_prediction)
print("Exact integral:", integral_value(p_tensor[0], torch.full((1,n_dim),1)[0]).item())

#Plotting loss just to see
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Derivative MSE loss")
plt.yscale("log")
plt.show()

''' To do:
- add better checking/replications of training
- change way points are generated (Xavier initialisation, 1D laticce)
- add pysecdec via wsl2 ubuntu
- Pick model version with lowest loss and save it to file/take average of multiple runs
- Run on GPU? Figure out how to do that if I am not already doing so'''