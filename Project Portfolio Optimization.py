
#Portef1.txt, RetRisk31_SolVal.txt, port1.txt ect. files saved to directory
import pandas as pd

df_un = pd.read_csv(" ...your properties...")
df_un.to_csv("portef1.csv", index=False)

print(df_un)

#%%
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pandas_datareader import data as pdr
from datetime import date
from pymoo.termination import get_termination
from pymoo.util.remote import Remote
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import pandas as pd
import numpy as np
from pymoo.util.remote import Remote
import seaborn as sns
import time
#%%
#cesarone dataset
file_name = 'RetRisk98_SolVal.csv'  # Replace 'RetRisk(32,85,89...)_SolVal.csv'
risk_column = 'Risk_K2'  # Replace 'Risk_K2,5,10'

data = pd.read_csv(file_name, delimiter='\s+', header=None)
data.columns = ['Return'] + [f'Risk_K{i}' for i in range(2, 12)] 
#%%
test_proplem = pd.read_fwf(
    'port4.csv', sep=" ",header=None) #port2.csv, port3.csv, port4.csv, port5.csv
print(test_proplem)

#def n as number of assets in dataset
n = int(test_proplem[0].iloc[0])

#split data into two data frames
test_proplem_RR = test_proplem.iloc[1:n+1]
test_proplem_CV = test_proplem.iloc[n+1:]
print(test_proplem_RR)

#split the single column into two other colums & give them headings each
test_proplem_RR['mean returns'] = [d.split()[0] for d in test_proplem_RR[0]]
test_proplem_RR['sd'] = [d.split()[1] for d in test_proplem_RR[0]]
del test_proplem_RR[0]
test_proplem_RR = test_proplem_RR.reset_index(drop=True)
print(test_proplem_RR)

#repeat but now split into three columns
test_proplem_CV['i'] = [d.split()[0] for d in test_proplem_CV[0]]
test_proplem_CV['j'] = [d.split()[1] for d in test_proplem_CV[0]]
test_proplem_CV['correlation'] = [d.split()[2] for d in test_proplem_CV[0]]
del test_proplem_CV[0]
test_proplem_CV = test_proplem_CV.reset_index(drop=True)
print(test_proplem_CV)

#convert variables to numeric variables
test_proplem_CV = test_proplem_CV.apply(pd.to_numeric)
test_proplem_RR = test_proplem_RR.apply(pd.to_numeric)
# correlation matrix
test_proplem2 = test_proplem_CV.pivot(index='i',
                                      columns='j', values='correlation')
test_proplem3 = np.triu(test_proplem2)
iu = np.triu_indices(n,1)
il = (iu[1],iu[0])
test_proplem3[il]=test_proplem3[iu]

test_proplem_std = np.asarray(test_proplem_RR['sd'])
# covariance matrix
test_proplem4 = np.multiply(test_proplem3, test_proplem_std)
test_proplem_cov = np.multiply(test_proplem4, test_proplem_std)

test_proplem_mr = np.asarray(test_proplem_RR['mean returns'])
#%%
unconstrained_ef = pd.read_fwf(
    'portef4.csv', sep=" ",header=None) # portef2.csv, portef3.csv, portef4.csv, portef5.csv

#assign the columns into two seperate dataframes
mean_return_un = unconstrained_ef[0]
variance_of_return_un = unconstrained_ef[1]

#name columns
unconstrained_ef = unconstrained_ef.rename(columns={0 : 'returns'})
unconstrained_ef = unconstrained_ef.rename(columns={1 : 'risk values'})

#swap columns
columns_titles1 = ["risk values","returns"]
unconstrained_ef = unconstrained_ef.reindex(columns=columns_titles1)

unconstrained_ef = np.asarray(unconstrained_ef)
#%%
import numpy as np

tol = 1e-3
v = test_proplem_cov
m = test_proplem_mr
#cardinality constraint
K = 5

CC = np.zeros((100, n))
for i in range(100):
    CC[i,:K] = np.random.uniform(0, 1, size = (K,))
    #print("CC = ", CC[i,:K])
    #nromalise K so sum = 1, then divide K elts by total their sum
    CC[i, :K] /= np.sum(CC[i, :K])
    #print(CC[i])
    # shuffle K elts randomly
    np.random.shuffle(CC[i, :K])
#%%
start = time.time()
class MyProblem(ElementwiseProblem): 
    def __init__(self, m, v):
        super().__init__(n_var=n,
                         n_obj=2,
                         n_constr=1,
                         xl=np.array([0 for _ in range(n)]),
                         xu=np.array([1 for _ in range(n)]))
        self.m = m
        self.v = v     
    def _evaluate(self, x, out, *args, **kwargs):
        #x is normalised
        x = x/np.sum(x)
        #risk of portfolio
        f1 = np.dot(np.array(x).T, (np.dot(self.v, np.array(x))))
        #maximise returns
        f2 = -(np.dot(np.array(x), self.m))
        #constraint: number of assets with non-zero weight should be K
        g2 = sum(i > tol for i in x)-K
        out["F"] = [f1,f2]
        out["G"] = [g2]
#%%
problem = MyProblem(m,v)
algorithm = NSGA2(pop_size=100, sampling=CC) 
termination = get_termination("n_gen", 8000) 
res = minimize(problem,
               algorithm,
               termination,
               seed=None,
               save_history=True,
               verbose=True)

X = res.X
sum_of_rows = X.sum(axis=1)
X = X / sum_of_rows[:, np.newaxis]
F = np.abs(res.F)

end = time.time()
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))
plt.scatter(variance_of_return_un, mean_return_un, s=4, facecolors='none', edgecolors='red', label='Beasley Dataset')
plt.scatter(F[:, 0], F[:, 1], s=15, facecolors='none', edgecolors='black', label='K=5')
plt.scatter(data[risk_column], data['Return'], s=2, facecolors='none', edgecolors='blue', label='Cesarone')
plt.title("Efficient Frontier")
plt.xlabel("Risk")
plt.ylabel("Expected Return")
plt.legend(loc='lower right', frameon=True)
plt.show()

#%%
print("Time taken to run algoritm", end - start)
from pymoo.indicators.igd import IGD
def _calc_pareto_front():
    un = unconstrained_ef
    return un

pf = _calc_pareto_front()
pf_un = pf[0]

true_pareto_front = data[['Return', risk_column]].values

# Calculate IGD
igd_calculator = IGD(true_pareto_front)
igd_value = igd_calculator(pf)

print("IGD value for the Caesarone Dataset:", igd_value)
#%%
