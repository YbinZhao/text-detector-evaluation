import numpy as np
from scipy.optimize import minimize
import jsonlines
import os
import json
import time
from utils import *
     
def create_irt_dataset(responses, dataset_name): 
    
    """
    Creates a dataset suitable for IRT analysis from a given set of responses and saves it in a JSON lines format.
    
    Parameters:
    - responses: A numpy array where each row represents a subject and each column a question.
    - dataset_name: The name of the file where the dataset will be saved.
    """
    
    dataset = []
    for i in range(responses.shape[0]):
        aux = {}
        aux_q = {}
        
        # Iterate over each question to create a response dict
        for j in range(responses.shape[1]):
            aux_q['q' + str(j)] = int(responses[i, j])
        aux['subject_id'] = str(i)
        aux['responses'] = aux_q
        dataset.append(aux)
    
    # Save the dataset in JSON lines format
    with jsonlines.open(dataset_name, mode='w') as writer:
        writer.write_all([dataset[i] for i in range(len(dataset))])

def train_irt_model(dataset_name, model_name, D, lr, epochs, device):
    
    # Constructing the command string
    command=f"py-irt train multidim_2pl {dataset_name} {model_name} --dims {D} --lr {lr} --epochs {epochs} --device {device} --priors hierarchical --seed 42 --deterministic --log-every 200"
    os.system(command)
        
def load_irt_parameters(model_name):

    with open(model_name+'best_parameters.json') as f:
        params = json.load(f)
    A = np.array(params['disc']).T[None, :, :]
    B = np.array(params['diff']).T[None, :, :]
    Theta = np.array(params['ability'])[:,:,None]
    return A, B, Theta



def estimate_ability_parameters(responses_test, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"):

    D = A.shape[1]
    
    # Define the negative log likelihood function
    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A, B).squeeze()
        log_likelihood = np.sum(responses_test * np.log(P + eps) + (1 - responses_test) * np.log(1 - P + eps))
        return -log_likelihood
    
    # Ensure the initial theta is a numpy array with the correct shape
    if type(theta_init) == np.ndarray:
        theta_init = theta_init.reshape(-1)
        assert theta_init.shape[0] == D
    else:
        theta_init = np.zeros(D)

    # Use the minimize function to find the ability parameters that minimize the negative log likelihood
    optimal_theta = minimize(neg_log_like, theta_init, method = optimizer).x[None,:,None] 
    
    return optimal_theta