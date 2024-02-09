# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:33:24 2024

@author: atrum
"""

import numpy as np

#def read_data(fname_A, fname_c):
    #A = np.loadtxt(fname_A)
    #b = 30
    # = np.loadtxt(fname_c)
    #return A,b,c
    
def init(n, m, perc):
    ''' More intuitive approach '''
    pop = np.zeros((n,m)) #n is number of rows (options) and m is number of columns (genes)
    pop[np.random.random(size=(n,m))<=perc] = 1 #pull uniformly between 0 and 1 for n x m
                                                #gives Trues when random value is less than perc
                                                #assigns True indices to 1 and False remains 0
    ''' Alternate (faster) approach '''
    #pop = np.random.choice(np.array([0,1]), size=(n,m), p=(1-perc,perc))
    return pop

def feasible(pop,c,budget):
    for i in range(pop.shape[0]): #for each population member 
        ''' randomly delete locations until within budget '''
        while pop[i]@c > budget: #pick a gene to turn to 0 until it is feasible
            ''' Randomly choose a "selected" location '''
            idx = np.random.choice(np.arange(pop.shape[1])*(pop[i]==1))
            pop[i,idx] = 0

def fitness(pop,A):
    return pop@A #returns number of autoplants are served

def select(pop, fit, method = 'rank linear'):
    ''' Compute probability distribution for choosing parents '''
    #higher the fitness, the higher probability we want to select them as parents
    if method == 'proportional':
        prob = fit/sum(fit) #fit is a numpy array of fitness for each pop member-> prob is array for all
    elif method == 'rank linear':
        r = np.zeros((n,))
        r[np.argsort(fit)] = np.arange(fit.shape[0])
        prob = (1 + r)/sum(1 + r)
    elif method == 'rank nonlinear':
        q = 0.7
        r = np.zeros((n,))
        r[np.argsort(fit)] = np.arange(fit.shape[0])
        prob = (q**(1 + r))/sum(q**(1 + r))
    parents = np.random.choice(np.arange(pop.shape[0]), size=(pop.shape[0],2), p = prob)
                #random.choice lets use choose between a subset (vector from 0 to max pop index)
                #we want to keep pop count constant but we need 2 parents 
                #prob a member is chosen as a parent is p = prob
    return parents

def crossover(parents,pop):
    crosspts = np.random.randint(0,pop.shape[1],size=pop.shape[0]) #which gene index should we merge parents on 
                                                                   #take beginning of first parent and end of second parent
    #idx = np.tile(np.arange(pop.shape[1]), pop.shape[0]).reshape(pop.shape[0],pop.shape[1])
    idx = np.tile(np.arange(pop.shape[1]), (pop.shape[0],1))
    par_left_mask = np.ones(pop.shape)*(idx<=crosspts.reshape(pop.shape[0],1))
    par_right_mask = np.ones(pop.shape)*(idx>crosspts.reshape(pop.shape[0],1))
    pop = pop[parents[:,0]]*par_left_mask + pop[parents[:,1]]*par_right_mask

def mutate(pop, perc):
    mut_idx = np.random.random(pop.shape)<=perc
    pop[mut_idx] = 1 - pop[mut_idx] #picks a gene to change from 0 to 1 or vice versa

def stat(pop, fit, best_fit, best_soln):
    if fit.max() > best_fit:
        return fit.max(), pop[fit.argmax()].copy()
    else:
        return best_fit, best_soln
    
    

def report(i, best_fit, fit):
    print(f'Generation {i}: Best fit: {best_fit}; Max fit gen:{fit.max()}; Avg fit gen: {fit.mean()}')
    
''' Input data '''
import mysql.connector as mySQL

mysql_user_name = 'root'
mysql_password = 'MySQL'
mysql_ip = '34.145.197.191'
mysql_db = 'cap_bud'

cnx = mySQL.connect(user=mysql_user_name, passwd=mysql_password,
                    host=mysql_ip, db=mysql_db)
cursor = cnx.cursor()
cursor.execute('SELECT * FROM data')
rows = cursor.fetchall()
data = []

for row in rows:
    data.append([row[1], row[1]*(1 + row[2])])
cursor.close()
cnx.close()

A = np.array([roi for _, roi in data])
c = np.array([cost for cost, _ in data])
budget = 500000

#A,budget,c = read_data('A.txt', 'c.txt')
num_loc = A.shape[0] #number of rows
#num_dest = A.shape[1] #number of cols

''' GA parameters '''
n = 500 # population size
num_gen = 100
init_perc = 0.05 # expected percentage of possible locations 
                 # selected in initial candidate solutions
mutate_perc = 0.001
prob_method = 'rank nonlinear'

''' Initialize population '''
pop = init(n, num_loc, init_perc)
feasible(pop,c,budget) # note that function modifies population directly
fit = fitness(pop,A)
best_fit, best_soln = stat(pop, fit, 0, np.zeros(num_loc))
plot_maxfit = np.zeros(num_gen)
plot_avgfit = np.zeros(num_gen)
plot_bestfit = np.zeros(num_gen)

''' Evolution '''
for i in range(num_gen):
    report(i, best_fit, fit) #best_fit is number of facilities served, increases over time with algorithm 
    parents = select(pop, fit, method= prob_method) #proportional, rank linear, rank nonlinear
    crossover(parents,pop) # replace population with offspring
    mutate(pop, mutate_perc)
    feasible(pop,c,budget)
    fit = fitness(pop,A)
    best_fit, best_soln = stat(pop, fit, best_fit, best_soln)
    plot_maxfit[i] = fit.max()
    plot_avgfit[i] = fit.mean()
    plot_bestfit[i] = best_fit
    
    
x = np.arange(0, num_gen, 1)

import matplotlib.pyplot as plt

plt.plot(x, plot_bestfit, label='Best Fit', color='blue')
plt.plot(x, plot_avgfit, label='Avg Fit', color='red')
plt.plot(x, plot_maxfit, label='Max Fit', color='green')

# Customize the plot
plt.title(f'Best Fit with {n} Generations: {best_fit} - Sort Method: {prob_method}')
plt.xlabel('Generation')
plt.ylabel('Investment Portfolio')
plt.legend()

# Display the plot
plt.show()
    