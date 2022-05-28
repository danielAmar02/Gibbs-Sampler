import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy
from scipy import stats
import tqdm
from tqdm import tqdm
from collections import Counter
import mpmath
import time
import seaborn as sns
from time import time
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import plot_confusion_matrix
import config


def initandfind_N_ujmanquantes(df,Lambda=1): 
    ntot=0
    df_modif=df.copy()
    for name_columns in df_modif.columns:
        if name_columns[0]=='J':
            for index_val in range(len(df_modif[name_columns])):
                if df_modif[name_columns][index_val]<0:
                    ntot+=1
                    df_modif[name_columns][index_val]=np.random.poisson(3)
    return df_modif,ntot


def init_alpha(K):
    return stats.dirichlet.rvs([0.5 for i in range(K)])[0]

def init_theta(K,J,a,b):
    return np.random.gamma(a,1/b,(K,J))

def pmf(liste,theta): #Cette fonction calcule le produit dans l'étape (*)
    res=1
    for i in range(len(liste)):
        res=res*stats.poisson.pmf(liste[i], theta[i])
    return res


def pmf_log(liste,theta): #Cette fonction calcule le produit dans l'étape (*)
    res=0
    for i in range(len(liste)):
        res=res+np.log(stats.poisson.pmf(liste[i], theta[i]))
    return res



def generation_deszi(Z,Proba):
        a=0
        b=Proba[0]
        k=1       
        aux=0
        for j in range(len(Proba)-1):
            if  Z >= a and  Z < b : 
                aux=k
            else : 
                k+=1
                a=a+Proba[j]
                b=b+Proba[j+1]
        return aux
    

def calcul_reach(df_true,df_pred):

    assert list(df_true.columns)==list(df_pred.columns)
    assert len(df_true)==len(df_pred)
    
    reach_true=0
    reach_pred=0
    
    
    for i in range(len(df_true)):
        
        if np.sum(df_true[df_true['users']==i+1][df_true.columns[1:]].values.tolist()[0])>0:
            reach_true+=1
            
        if np.sum(df_pred[df_pred['users']==i+1][df_pred.columns[1:]].values.tolist()[0])>0:
            reach_pred+=1
    
    return reach_true/len(df_true), reach_pred/len(df_true)

def init(K,J,N):
    reach={}
    reach['Gibbs']=[0 for i in range(N)]
    reach['Bench']=[0 for i in range(N)]
    reach['Bench1']=[0 for i in range(N)]
    Val_Prises={}
    
    for n in range(N):
        Val_Prises[n]=[]
    Nmax=[0 for i in range(N)]
    Nitier=[i+1 for i in range(N)]
    Number_of_right_answer={}
    Number_of_right_answer['Gibbs']=[0 for i in range(N)]
    Number_of_right_answer['Bench']=[0 for i in range(N)]
    ecart=[0 for i in range(N)]
    #ecart['Gibbs']=[0 for i in range(N)]
    #ecart['Benchmark']=[0 for i in range(N)]
        
    ecart_df={}
        
    ecart_df['Gibbs']=[0 for i in range(N)]
    ecart_df['Bench']=[0 for i in range(N)]
        
    theta_convergence=[0 for i in range(N)]
    alpha_convergence=[0 for i in range(N)]
    trace_matrix_theta=np.zeros((K,J,N))
    trace_matrix_alpha=np.zeros((K,N))
    
    return reach,Val_Prises,Nmax,Nitier,Number_of_right_answer,ecart,ecart_df,theta_convergence,alpha_convergence,trace_matrix_theta,trace_matrix_alpha
    
            
        
def update_z_and_generate_Z(alpha,Nuj,theta,P,K,J):
    z=np.zeros((P,K))
    Z=[0 for i in range(P)]
    for u in range(P):
                total=0
                for k in range(K):
                    
                    thetaaux=theta[k,:]
                    #z[u,k]= alpha[k]*pmf(Nuj[Nuj['users']==u+1][['J1','J2','J3']].values.tolist()[0],thetaaux)
                
                    z[u,k]= np.log(alpha[k])+pmf_log(Nuj[Nuj['users']==u+1][['J' + str(j+1) for j in range(J)]].values.tolist()[0],thetaaux)

                    #total+= z[u,k]
               
                max_zu=np.max(z[u,:])    
                total=max_zu+scipy.special.logsumexp(z[u,:]-max_zu)

                for k in range(K):
                    
                        z[u,k]=mpmath.exp(z[u,k]-total)
                        #z[u,k]=z[u,k]/total
                Z[u]=np.random.choice([k for k in range(K)],p=z[u,:],size=1)[0]
                
    return z,Z





def generate_q(Z,beta,K):
    
    
    n_count=[0 for i in range(K)]
    concentration_parameters=[0 for i in range(K)]
    
    for i in Counter(Z):
        n_count[i]=Counter(Z)[i]

    for k in range(K):
        concentration_parameters[k]=beta+n_count[k]

    return n_count,stats.dirichlet.rvs(concentration_parameters)[0] #Génération des alphas
                
            
        
def generate_theta(K,J,P,Z,n_count,Nuj,a,b):
        res=np.zeros((K,J))
        aux=np.zeros((K,J)) 
        for u in range(P):
            for k in range(K):
                if Z[u]==k:
                    for j in range(J):
                        aux[k,j]+=Nuj[Nuj['users']==u+1]['J'+str(j+1)].tolist()[0]

        for k in range(K):
            for j in range(J):
                res[k,j]=np.random.gamma(a+aux[k,j],1/(b+n_count[k]))
        return res


def plot_cm(y_true, y_pred, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "YlOrRd", annot=annot, fmt='', ax=ax)
    


# Reach par classe
# Taux de duplication
def analyse_matrice(Nuj_pred_numpy,Nuj_true_numpy,Z_true,K):
    
    d={}
    reach_pred=0
    reach_true=0
    d['Dataset']=['True','Predicted']

    for k in range(K):
      d['Reach_' +str(k)]=[0,0]


    for k in range(K):
        d["Frequency_K=" +str(k)]=[0,0]



    assert len(Nuj_pred_numpy)==len(Nuj_true_numpy)
    P=len(Nuj_true_numpy)

    for u in range(P):
              if np.sum(Nuj_pred_numpy[u][1:])>0:
                reach_pred+= 1.0/P
                d['Reach_' +str(Z_true[u])][1]+=1/P
              if np.sum(Nuj_true_numpy[u][1:])>0:
                reach_true+= 1.0/P
                d['Reach_' +str(Z_true[u])][0]+=1/P

    d['Reach']=[reach_true,reach_pred]


    for u in range(P):
      d["Frequency_K=" +str(Z_true[u])][0] +=(1/(P*d['Reach_' +str(Z_true[u])][0]))*np.sum(Nuj_true_numpy[u][1:])
      d["Frequency_K=" +str(Z_true[u])][1] +=(1/(P*d['Reach_' +str(Z_true[u])][0]))*np.sum(Nuj_pred_numpy[u][1:])

      



    return pd.DataFrame.from_dict(d)


    





