import src/config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import seaborn as sns
from time import time

## Creation of a dataset with correlations among devices

def dataset_LCM(P,J,K,prop_bi,prop_sum,MoyLCM,Mass0,hidden_parameter,deseq_param=True):
    """
    deseq_param=True : On choisit les poids des diracs en 0 comme Mass0*uniforme entre 0 et 1 
    deseq_param=False : On choisit les poids des diracs en 0 tous égaux à Mass0
    """
    
    
    assert hidden_parameter <= J, 'The number of hidden parameters must be lower than the number of parameters'
    assert prop_bi+prop_sum <= 1, 'Sum of proprotions must be lower than 1'
    Frobenius_norm=np.zeros((K,K))
    users=[i for i in range(1,P+1)]
    d={}
    d['users']=users
    Sigma_matrix={}
    alpha={}
    theta=np.zeros((K,J))
    Z=[0 for i in range(P)]
    
    for i in range(J):
        d['J'+str(i+1)]=[]    
    
        
    if deseq_param:
        poids_0=Mass0*np.random.rand(K,J)

        for i in range(K):
            A = np.random.rand(J,J)
            B = np.dot(A,A.transpose())
            Sigma=B
            Sigma_matrix[i]=Sigma
            alpha[i]=np.random.multivariate_normal(np.array([MoyLCM for m in range(J)]),Sigma)
        
        for k0 in range(K):
            for k1 in range(K):
                Frobenius_norm[k0][k1]=np.linalg.norm(Sigma_matrix[k0]-Sigma_matrix[k1])
            
            
        for i in range(len(users)):
            k=np.random.randint(0,K)
            Z[i]=k
            for j in range(J):
                if np.random.rand()<=poids_0[k][j]:
                    d['J'+str(j+1)]+=[0]
                else:
                    d['J'+str(j+1)]+=[np.random.poisson(np.exp(alpha[k][j]))]
                theta[k][j]=np.exp(alpha[k][j])

        ax = plt.axes()
        sns.heatmap(Frobenius_norm, ax = ax,cmap="YlGnBu",annot=True)

        ax.set_title('Matrice des normes de Frobenius')
        plt.show()
        
    else :
        poids_0=Mass0*np.ones((K,J))
        liste_theta=2*np.random.rand(K,J)
        #assert K==2, "K must be equal to 2"
        #assert J==2, "J must be equal to 2"
        #for i in range(K):
         #   A = np.random.rand(J,J)
          #  B = np.dot(A,A.transpose())
          #  Sigma=B
           # Sigma_matrix[i]=Sigma
           # alpha[i]=np.random.multivariate_normal(np.array([12 for m in range(J)]),Sigma)
        
      #  for k0 in range(K):
      #      for k1 in range(K):
       #         Frobenius_norm[k0][k1]=np.linalg.norm(Sigma_matrix[k0]-Sigma_matrix[k1])
                
        
            
        
        Z=np.random.binomial(K-1, 0.5, P)
        for i in range(len(users)):
            for k in range(K):
                if Z[i]==k:
                    for j in range(J):
                        
                        if np.random.rand()<=poids_0[k][j]:
                            d['J'+str(j+1)]+=[0]
                        else:
                            #d['J'+str(j+1)]+=[np.random.poisson(np.exp(alpha[k][j]))]
                            d['J'+str(j+1)]+=[np.random.poisson(liste_theta[k][j])]
        theta=np.array(liste_theta)
                        #theta[k][j]=np.exp(alpha[k][j])
                        
    



    alpha_proportion=[0 for i in range(K)]
    
    for i in Counter(Z):
        alpha_proportion[i]=Counter(Z)[i]/len(Z)
    alpha_proportion=np.array(alpha_proportion)

    df_sans_independance=pd.DataFrame.from_dict(d)
    df_sans_independance=df_sans_independance[['users']+['J' +str(j+1) for j in range(J)]]
    df_sans_independance_partial=df_sans_independance.copy()
    df_sum=df_sans_independance.copy()
    #S=[df_sans_independance[column].sum() for column in df_sans_independance.columns[1:]]
    S=[0 for i in range(J)]

    for index_users in range(P):
        ## 2 devices cachés 
        if index_users/P<=prop_bi:
            M=random.sample([i for i in range(J)], hidden_parameter)
            for m in M:
                df_sans_independance_partial.loc[df_sans_independance_partial['users']==index_users+1,'J'+str(m+1)]=-1
                df_sum.loc[df_sum['users']==index_users+1,'J'+str(m+1)]=-1
        elif prop_bi<index_users/P<=prop_sum+prop_bi:
            for m in range(J):
                df_sum.loc[df_sum['users']==index_users+1,'J'+str(m+1)]=-2
                S[m]+=list(df_sans_independance[df_sans_independance['users']==index_users+1]['J'+str(m+1)])[0]
                

        ### 1 seul devices caché    
        #if index_users/P<=prop_bi:
         #   c_u = np.random.binomial(1, 0.4, J)
          #  aux=np.sum(c_u)
           # while aux==0 or aux==J or aux==1:
            #    c_u = np.random.binomial(1, 0.4, J)
             #   aux=np.sum(c_u)

            #for index_cuj in range(len(c_u)):
             #   if c_u[index_cuj]==0:
              #      df_sans_independance_partial.loc[df_sans_independance_partial['users']==index_users,'J'+str(index_cuj+1)]=-1
        #else:
         #   break
    #print(f'le vrai theta est : \n {theta} \n')
    #print(f'les proprortions des classes K sont {Counter(Z)}')
    #print(alpha_proportion)
    
    return df_sans_independance,df_sans_independance_partial,alpha_proportion,theta,df_sum,S,Z
