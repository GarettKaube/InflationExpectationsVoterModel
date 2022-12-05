from random_graph_and_relation_matrix import Graph
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import pickle
from sklearn.linear_model import ridge_regression
import os


parser = argparse.ArgumentParser(description='Arguments for voter model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--population", help="Number of nodes/individuals in model", default=10)
parser.add_argument("--edges", help="Number of edges for random graph, n>=m", default=20)
parser.add_argument("--data",help="r or p where r=random initial expectations, p=predifined data in text file.", default='rn')
parser.add_argument("--num_runs",help="number of runs for the voter model", default='10')
parser.add_argument("--torch",help="use torch model" , default=False)
parser.add_argument("--iterations",help="number of iterations per votermodel" , default=500)

# this is here so we can load a regression neural network for the voter model 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)



class VoterModel():
    def __init__(self, n, m):
        self.n = n          # height of matrix
        self.m = m          # number of edges
        self.graph = Graph(n,m) # random graph


    def get_initial_expectations(self):
        """open predefined initial expectations text file
        Returns: 
            array of initial expectations
        """

        with open('data.txt', 'r') as f:
            initial_expectations = np.array([float(i) for i in f.read().split('\n')])
            assert len(initial_expectations) == self.n, "Not enough inflation expectations"

        return initial_expectations


    def random_initial_expectations(self, dist):
        """Generates random normal(2, 0.4) inflation expectations
        Returns: 
            array of initial expectations
        """
        if dist == 'rn':
            random_expectations = np.random.normal(2, 0.4, size = self.n)
        elif dist =='rexp':
            random_expectations = 8-np.random.exponential(2, size = self.n)
        else:
            raise ValueError("Invalid input for data")
        return random_expectations


    def Voter_Model(self, init_expectations):
        """Voter model
        Input:
            init_expectations: array of initial expectations
        Returns:
            init_expectations_copy: array
        """
        # create a copy of inflation expectations 
        init_expectations_copy = init_expectations.copy()
        # get relation matrix from random graph
        matrix = np.array(self.graph.get_relation_matrix())
        # adjust expectations
        for i in range(self.n):
            neighbors = np.where(matrix[i] == 1)
            new_expectation = self.adjust_expectation(i, neighbors, init_expectations)
            init_expectations_copy[i] = new_expectation
            
        return init_expectations_copy


    def adjust_expectation(self, person, neighbors, init_expectations):   
        """Adjusts individuals inflation expectation
        Input:
            person : int of row of relation matrix
            neighbor: array of integers corresponding to neighbors column in relation matrix
        Returns:
            persons_expectation: updated inflation expectation for person
        """
        persons_expectation = init_expectations[person]
        neighbors_expectations = init_expectations[neighbors] # array of neighbors expectations
        persons_expectation = np.mean(neighbors_expectations)
        for i in range(len(neighbors)):
            p = 0.7
            #increase probability of expectations going up if expectations are below zero
            if persons_expectation < 0:
                p = 0.9
            
            # individuals neighbor influences their inflation expectation to change with p = 0.7 to add random normal and 1-p=0.3 to subtract random normal from original expectation
            binn = np.random.binomial(1, p, size = 1)[0] 
            randn = np.random.normal(0, 0.4115) # random normal for adjusting expectation
            if binn == 1:      
                persons_expectation = persons_expectation + randn
            else:
                persons_expectation = persons_expectation - randn
                
        return persons_expectation


    def predict_inflation(self, expectation, model = None, torch_ = False):
        """Regression model for predicting inflation in 5 years
        Input:
            expectation: array of inflation expectations
            model: torch model or sklearn model for prediction
            torch: bool for if torch is being used
        Return: 
            array of predicted inflation
        """
        if torch_ == True:
            # torch model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = model.to(device).double()
            model.eval()
            expectation_tensor = torch.from_numpy(expectation.reshape(-1,1).astype(float)).to(device)
            predicted_inflation = model(expectation_tensor)

        else:
            # predict with sklearn model
            expectation = expectation.reshape(-1,1)
            predicted_inflation = model.predict(expectation)

        return predicted_inflation


    def export_relation_matrix(self, i):
        """
        Generates a text file of the random relation matrix
        Input:
            i: voter model run i
        return: None
        """

        rm = np.array(self.graph.matrix).astype(int)
        with open(f'./relation_matrices/relation_matrix{i}.txt',"w") as f:
            np.savetxt(f, rm, fmt = '%i')


def save_plot(expectations, iterations, predictions, i):
    """Plots mean of expected inflation generated from voter model and save
    Input:
        expectations: array of floats of inflation expectations
        iterations: array storing iterations/time that the voter model has gone through
        predictions: array of predictions
    """
    fig = plt.figure()
    ax = fig.add_axes([1,1,1,1])
    x = np.arange(iterations)
    plt.plot(x, expectations, label = f'Inflation Expectations for run{i}')
    plt.plot(x, predictions, label = f'Predicted Inflation for run{i}')
    plt.xlabel('Time/Iterations')
    plt.ylabel('Inflation')
    plt.legend()
    plt.savefig(f'./votermodelplots/vm_fig{i}',bbox_inches='tight')
    

def main():
    args = parser.parse_args()
    print("Population:", args.population)
    inp = args.data                         # random data or predefined 
    runs = int(args.num_runs)               # number of runs of the votermodel
    n = int(args.population)                # number of individuals
    m = int(args.edges)                     # must be >= than n
    torch_ = bool(args.torch)
    plot_every = 10                         #plot mean inflation expectations every couple of runs, change if you want more plots
    num_of_iterations = int(args.iterations)                # how long each voter model will run

    average_expectations_per_run = []       # keep track of mean inflation expectations per run of the voter model
    all_average_inflation_expectations = []
    arr = np.empty((runs,num_of_iterations)) # table for keeping track of inflation expectations across all runs and iterations for plotting purposes 
    

     
    
    if torch_ == False:
        # polynomial degree 5 regression with l2 penalty lambda = 0.1
        loaded_model = pickle.load(open('./models/ridge_reg_model.sav', 'rb'))
    else:
        loaded_model = torch.load(r'./models/neural_2layers_sigmoid3.pth')
    

    print('Creating directory for relation matrices and plots.')

    try: 
        os.mkdir('./relation_matrices') 
    except OSError as error:
        print('relation_matrices directory already exists') 

    try:
        os.mkdir('./votermodelplots') 
    except OSError as error:
        print('votermodelplots directory already exists') 
          
    
    for i in range(runs):
        voter = VoterModel(n,m) # create new random graph for run i
        
        # get initial expectations
        if inp == 'p':
            init_expectations = voter.get_initial_expectations()
        else:
            init_expectations = voter.random_initial_expectations(inp)

        print("Initial expectations:",  '\n' + '_'*100 + '\n', init_expectations, '\n' + '='*100 + '\n' + '_'*100)


        expectations_over_iterations = []  # keep track of mean inflation expectations over time
        predictions = []                   # keep track of predictons 
             

        print('Calculating new inflation expectations...')

        # run voter model
        for j in range(num_of_iterations):
            init_expectations = voter.Voter_Model(init_expectations)
            mean_expectation = np.mean(init_expectations)
            expectations_over_iterations.append(mean_expectation)
            all_average_inflation_expectations.append(mean_expectation)
            if torch_ == True:
                predictions.append(voter.predict_inflation(mean_expectation, loaded_model, torch_).detach().cpu().numpy()[0][0])
            else: 
                predictions.append(voter.predict_inflation(mean_expectation, loaded_model))

            arr[i,j] = mean_expectation     # add inflation expectation to collection 

        # create relation matrices
        voter.export_relation_matrix(i)
        print(f"Final Inflation Expectations for run {i}:\n {init_expectations}")
        mean = np.mean(expectations_over_iterations)
        std = np.std(expectations_over_iterations)
       
        print(f'Mean of inflation expectations for run {i}: {mean} \nStandard deviation for run {i}: {std}.')
        
        print(i)
        if i % plot_every == 0:
            # plot results

            save_plot(expectations_over_iterations, num_of_iterations, predictions, i)

    # plot average inflation expectation for each graph at iteration i
    averages = arr.mean(axis=0)
    fig4 = plt.figure()
    ax = fig4.add_axes([1,1,1,1])
    plt.plot(range(num_of_iterations), averages, label = 'Inflation expectation') # average inflation expectations for each graph
    plt.xlabel('iterations')
    plt.ylabel('Inflation')
    plt.legend()
    plt.savefig('./votermodelplots/mean_per_it', bbox_inches='tight')

        



if __name__ == '__main__':
    main()
