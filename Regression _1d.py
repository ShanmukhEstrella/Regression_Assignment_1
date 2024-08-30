#Shanmukh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import math

def phigenerator(phi,x):
  N = len(phi)
  D = len(phi[0])
  for i in range(N):
    for j in range(D):
      phi[i][j] = x[i] ** j

def predict(weights, x):
  value=0
  for i in range(len(weights)):
    value = value + weights[i] * (x ** i)
  return value

df = pd.read_csv('Testing_1d.csv')
x_testing = df['X'].tolist()
t_testing = df['y'].tolist()

df = pd.read_csv('Training_1d.csv')
x_training = df['X'].tolist()
t_training = df['y'].tolist()

df = pd.read_csv('Validation_1d.csv')
x_validation = df['X'].tolist()
t_validation = df['y'].tolist()

M_values = []
Erms_training=[]
Erms_validation=[]
M_req = 1
min_error = 10000
for M in [3,6,9]:
  D = M+1
  phi_training = np.zeros((len(x_training), D))
  phigenerator(phi_training,x_training)
  target_training = np.array(t_training)
  target_training.reshape(-1,1)
  minomega_vector = ((np.linalg.inv((phi_training.T) @ phi_training)) @ (phi_training.T)) @ target_training
  minerror_training = 0.5*((((phi_training@minomega_vector) - target_training).T) @ ((phi_training@minomega_vector) - target_training))
  prediction = [predict(minomega_vector,x) for x in x_training]
  target_validation = np.array(t_validation)
  target_validation.reshape(-1,1)
  phi_validation = np.zeros((len(x_validation), D))
  phigenerator(phi_validation,x_validation)
  minerror_validation = 0.5*((((phi_validation@minomega_vector) - target_validation).T) @ ((phi_validation@minomega_vector) - target_validation))
  M_values.append(M)
  Erms_training.append(minerror_training)
  Erms_validation.append(minerror_validation)
  if(min_error > minerror_validation):
    min_error = minerror_validation
    M_req = M
    min_omega = minomega_vector
    phi = phi_training
  # if(M==9):
  #   min_error = minerror_validation
  #   M_req = M
  #   min_omega = minomega_vector
  #   phi = phi_training
  # x_t = np.linspace(min(x_training),max(x_training),1000)
  # y_t = [predict(minomega_vector,z) for z in x_t]
  # plt.scatter(x_training, t_training ,label = "target_training",color='blue')
  # plt.plot(x_t, y_t ,label='non-regularised prediction',color='red')
  # plt.xlabel("x")
  # plt.ylabel("y")
  # plt.legend()
  # plt.show()
D_req = M_req + 1
print(M_req)

# plt.plot(M_values, Erms_training, label = "Training", color='blue',marker='o')
# plt.plot(M_values,Erms_validation, label = "Validation",color='red',marker='o')
# plt.xlabel('M')
# plt.ylabel('Erms')
# plt.title("Graph between M values and Erms")
# plt.legend()
# plt.show()

#testing_prediction = [predict(min_omega,x) for x in x_testing]
x_t = np.linspace(min(x_training),max(x_training),1000)
y_t = [predict(min_omega,z) for z in x_t]
I = np.eye(D_req)

# lamda = (1)
# min_omega_regularisation = ((np.linalg.inv((phi.T) @ phi) + (lamda)*(I)) @ (phi.T)) @ target_training
# extra_term = ((lamda//2) * (min_omega_regularisation.T @ min_omega_regularisation)).item()
# y_t_r = [extra_term + predict(min_omega_regularisation,z) for z in x_t]
plt.scatter(t_training, [predict(min_omega,z) for z in x_training] ,label = "tn vs y of training data",color='blue')

# plt.plot(x_t, y_t ,label='',color='red')
# plt.plot(x_t, y_t_r ,label='regularised prediction',color='green')
plt.plot()
plt.xlabel('tn')
plt.ylabel('y')
plt.title("Training t vs y ")
plt.xscale('linear')
plt.yscale('linear')
plt.legend()
plt.show()