#Shanmukh
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import random
import math

def predict(weights,x,m):
  z =  generate(m,x)
  z = np.array(z)
  z = z.reshape(-1,1)
  y = np.array(weights).reshape(-1,1)
  l = np.transpose(z) @ weights
  return l

def get_column_names(file_path):
    df = pd.read_csv(file_path)
    column_names = df.columns.tolist()
    return column_names

def generate(M,x):
   lst =[]
   for i in range(M+1):
     combination = combinations_with_replacement(x,i)
     s = [np.prod(comb) for comb in combination]
     lst = lst + s
   return lst

def phigenerator(M,phi,x):
  N = len(phi)
  for i in range(N):
    phi[i] = generate(M,x[i])

df = pd.read_csv('Testing_2d.csv')
file_path = 'Testing_2d.csv'
columns = get_column_names(file_path)
columns = columns[0:len(columns)-1]
x_testing = df[columns].values.tolist()
t_testing = df['output'].tolist()

df = pd.read_csv('Training_2d.csv')
file_path = 'Training_2d.csv'
columns = get_column_names(file_path)
columns = columns[0:len(columns)-1]
x_training = df[columns].values.tolist()
t_training = df['output'].tolist()

df = pd.read_csv('Validation_2d.csv')
file_path = 'Validation_2d.csv'
columns = get_column_names(file_path)
columns = columns[0:len(columns)-1]
x_validation = df[columns].values.tolist()
t_validation = df['output'].tolist()

M_values = []
Erms_training=[]
Erms_validation=[]
minierror = 10000000000000
M_req = 1
for M in [2,4,6]:
  D = ((M+2)*(M+1))//2
  phi_training = np.zeros((len(x_training), D),dtype=np.float64)
  phigenerator(M,phi_training,x_training)
  target_training = np.array(t_training)
  target_training.reshape(-1,1)
  minomega_vector = ((np.linalg.inv((phi_training.T) @ phi_training)) @ (phi_training.T)) @ target_training

  target_validation = np.array(t_validation)
  target_validation.reshape(-1,1)
  phi_validation = np.zeros((len(x_validation), D))
  phigenerator(M,phi_validation,x_validation)
  minerror_validation = 0.5*((((phi_validation@minomega_vector) - target_validation).T) @ ((phi_validation@minomega_vector) - target_validation))
  minerror_training = 0.5*((((phi_training@minomega_vector) - target_training).T) @ ((phi_training@minomega_vector) - target_training))
  if(minierror >= minerror_validation):
     minierror = minerror_validation
     M_req = M
     min_omega = minomega_vector
     phi = phi_training
  # if(M==6):
  #    minierror = minerror_validation
  #    M_req = M
  #    min_omega = minomega_vector
  #    phi = phi_training
  M_values.append(M)
  Erms_training.append(minerror_training)
  Erms_validation.append(minerror_validation)
  x1_training=[]
  for i in x_training:
    x1_training.append(i[0])
  x2_training=[]
  for i in x_training:
     x2_training.append(i[1])
  x1_validation=[]
  for i in x_validation:
    x1_validation.append(i[0])
  x2_validation=[]
  for i in x_validation:
     x2_validation.append(i[1])
  x1_testing=[]
  for i in x_testing:
    x1_testing.append(i[0])
  x2_testing=[]
  for i in x_testing:
     x2_testing.append(i[1])
  # x = np.linspace(min(x1_training),max(x1_training),100)
  # y = np.linspace(min(x2_training),max(x2_training),100)
  # X,Y = np.meshgrid(x, y)
  # Z = np.vectorize(lambda x, y: predict(minomega_vector, [x, y], M))(X, Y)
  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  # ax.plot_surface(X, Y, Z, color='yellow')
  # ax.scatter(x1_training,x2_training,t_training,color="blue")
  # if(M==4):
  #   fig = plt.figure()
  #   ax = fig.add_subplot(111, projection='3d')
  #   ax.plot_trisurf(x1_training, x2_training, t_training, color='blue', edgecolor='none')
  #   ax.scatter(x1_validation, x2_validation, t_validation, color='red', edgecolor='none')
  #   ax.set_xlabel('x1')
  #   ax.set_ylabel('x2')
  #   ax.set_zlabel('Predicted Y')
  #   plt.show()

D_req = ((M_req+2)*(M_req+1))//2
print(M_req)
phi_testing = np.zeros((len(x_testing),((M_req+2)*(M_req+1))//2),dtype=np.float64)
phigenerator(M_req,phi_testing,x_testing)
training_prediction = phi @ min_omega
training_prediction.reshape(1,-1)
plt.scatter(t_training,training_prediction,label = 'tn vs y of training data',color='blue')
plt.xlabel("tn")
plt.ylabel('y')
plt.legend()
plt.show()

# plt.plot(M_values, Erms_training, label = "Training", color='blue',marker='o')
# plt.plot(M_values,Erms_validation, label = "Validation",color='red',marker='o')
# plt.xlabel('M')
# plt.ylabel('Erms')
# plt.title("Graph between M values and Erms")
# plt.legend()
# plt.show()

# I = np.eye(D_req)
# lamda = (0.001)
# min_omega_regularisation = ((np.linalg.inv((phi.T) @ phi) + (lamda)*(I)) @ (phi.T)) @ target_training
# extra_term = ((lamda//2) * (min_omega_regularisation.T @ min_omega_regularisation)).item()

# x = np.linspace(min(x1_training),max(x1_training),100)
# y = np.linspace(min(x2_training),max(x2_training),100)
# X,Y = np.meshgrid(x, y)
# Z_reg = np.vectorize(lambda x, y: predict(min_omega_regularisation, [x, y], M_req))(X, Y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z_reg, color='yellow')
# ax.scatter(x1_training,x2_training,t_training,color="blue")
# plt.show()

# Z1 = [predict(min_omega,x,M_req) for x in x_testing]
# ax.set_xlabel('X1')
# ax.set_ylabel('X2')
# ax.set_zlabel('Predicted Y')
# ax.scatter(x1_testing,x2_testing,t_testing,color="blue")
# plt.show()