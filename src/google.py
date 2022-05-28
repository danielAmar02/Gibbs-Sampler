import src/config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import seaborn as sns
from time import time



def ADM(Tableau,L,sigma,N,J,P,prop_panel,N_tableaux):

    decimal=1
    while P/10**decimal>=10:
      decimal+=1

    res=preprocess_dataset(Tableau,prop_panel,P)

    T_panel=[np.array([res[n][1]]) for n in range(len(res))]
    T=np.reshape(T_panel,(N_tableaux,J))
    T=T/(P*prop_panel)

    R=[np.array([res[n][3]]) for n in range(len(res))]

    

  

    K=1
    K_liste=[1]
    alpha=[1]
    X=[[1 for i in range(J)]]

    for i in tqdm(range(N)):
      mus=X.copy()
      for m in range(L):
        covs = [sigma**2*np.eye(J) for i in range(len(X))]
        pis = alpha.copy()
        acc_pis = [np.sum(pis[:i]) for i in range(1, len(pis)+1)]
        assert np.isclose(acc_pis[-1], 1)
        # sample uniform
        r = np.random.uniform(0, 1)
        # select Gaussian
        k = 0
        for i, threshold in enumerate(acc_pis):
            if r < threshold:
                k = i
                break
        selected_mu = mus[k]
        selected_cov = covs[k]
        samples=np.random.multivariate_normal(selected_mu,selected_cov)
        print(samples)
        X+=[samples]
        
      K=K+L

      def objective(alpha2):
        aux=0
        for i in range(N_tableaux):
          S_aux=0
          
          for k in range(K):
            
            S_aux+=alpha2[k]* (1-exp(-np.vdot(X[k],T[i])))
          aux+=(R[i]-S_aux)**2
        return aux

      def contrainte_alpha1(alpha2):
        return -1+np.sum(alpha2)
      
      cons=({"type" : "eq", "fun" :  contrainte_alpha1 })

      alpha_initial_guess=stats.dirichlet.rvs([0.5 for i in range(K)])[0]
      

      sol= minimize(objective,alpha_initial_guess,method='slsqp',bounds=[(0,1) for i in range(K)],constraints=cons)

      alpha=sol.x
      for i in range(len(alpha)):
        
        if alpha[i]<10**(-decimal):
          alpha[i]==0
     

      new_X=[]
      new_alpha=[]
      for k in range(K):
        if alpha[k]!=0:
          new_alpha+=[alpha[k]]
          new_X+=[X[k]]

      K=len(new_alpha)
      K_liste+=[K]
      alpha=new_alpha.copy()
      X=new_X.copy()


      
    return K,alpha,X


def VID_modele(K,alpha,X,res,P,n):
  J=len(res[0][1])
  V={}
  users_reached=[]
  for k in range(K):
    V[k]=[i for i in range(int(np.sum(alpha[:k])*P),int(np.sum(alpha[:k+1])*P))]
    

  kappa=[]
  for j in range(J):
    aux=0
    for k in range(K):
      
      
      aux+=alpha[k]*X[k][j]
    kappa+=[aux]


  

  
  proba_classe={}
  for j in range(J):
    proba_classe[j]=[]
    for k in range(K):
       proba_classe[j]+=[(alpha[k]*X[k][j])/kappa[j]]

  for j in range(J):

    for e in range(res[n][2][j]):
      if np.random.binomial(1,np.min([kappa[j],0.9999]))==1:
        
        k= np.random.choice(np.arange(0,K), p = proba_classe[j], size = 1, replace = True)[0]
        
        VP_indice=np.random.randint(0,len(V[k]))
        users_reached+=[V[k][VP_indice]]

  return len(set(users_reached))/P








