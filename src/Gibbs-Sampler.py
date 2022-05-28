import config
import google
import utils

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



def algo_Isum(K,J,P,a,b,N,beta,M,prop_bi,prop_S,MoyLCM,Mass0,hidden_parameter,deseq_param=True,Lambda0=1):
    assert prop_S>0, "We do not know any information on the Nuj"
    NMI=[0 for i in range(N)]
    
    NMI_bench=[0 for i in range(N)]
    
    reach,Val_Prises,Nmax,Nitier,Number_of_right_answer,ecart,ecart_df,theta_convergenceinf,alpha_convergenceinf,trace_matrix_theta,trace_matrix_alpha=init(K,J,N)
    reach['Para']=[0 for i in range(N)]
    Classes={}
    sd={}
    sd['Gibbs']=np.zeros((N,M))
    res=[0 for i in range(N)]
    Saux={}

    
    for m in tqdm(range(M), desc='Nombre de simulations du Gibbs',position=0):
        
        
        
        df_sans_independance,df_sans_independance_partial,Truealpha,Truetheta,df_sum,S,Z_true = dataset_LCM(P,J,K,prop_bi,prop_S,MoyLCM,Mass0,hidden_parameter,deseq_param)
        print("Le vrai alpha est {}".format(Truealpha))
        Isum=list(df_sum[df_sum['J1']==-2]['users'])
        

        Nuj,ntot=initandfind_N_ujmanquantes(df_sum,Lambda0)
        Nuj_para,ntot_para=initandfind_N_ujmanquantes(df_sum,Lambda0)
        Nuj_para=Nuj_para[~Nuj_para.users.isin(Isum)]
        
        
        df_0_benchmark=df_sans_independance.copy()
        df_1_benchmark=df_sans_independance.copy()
        df_0_benchmark_numpy=df_0_benchmark.to_numpy() 
        df_1_benchmark_numpy=df_1_benchmark.to_numpy()
        df_sum_numpy=df_sum.to_numpy() 

        alpha=init_alpha(K)
        theta=init_theta(K,J,a,b)

        alpha_para=init_alpha(K)
        theta_para=init_theta(K,J,a,b)

        P_para=P-len(Isum)

        assert np.shape(theta)==np.shape(Truetheta)
        assert np.shape(alpha)==np.shape(Truealpha)

        reach_true,aux=calcul_reach(df_sans_independance,df_0_benchmark)
        print('Le vrai reach est {}'.format(reach_true))
        users=np.array([i for i in range(P) if i+1 not in Isum ])

       

        for n in tqdm(range(N)):

            #t0=time()  
            #reach_true,reach_pred=calcul_reach(df_sans_independance,Nuj)
            #print(time()-t0)
            

            
            z,Z=update_z_and_generate_Z(alpha,Nuj,theta,P,K,J)
            z_para,Z_para=update_z_and_generate_Z_para(alpha_para,Nuj_para,theta_para,P_para,K,J,users)


            NMI[n]+=1/M*normalized_mutual_info_score(Z_true,Z)
            NMI_bench[n]+=1/M*normalized_mutual_info_score(Z_true,np.random.randint(K,size=len(Z)))


            
            n_count,alpha=generate_q(Z,beta,K)
            alpha=np.array(alpha)
            theta=generate_theta(K,J,P,Z,n_count,Nuj,a,b)

            n_count_para,alpha_para=generate_q(Z_para,beta,K)
            alpha_para=np.array(alpha_para)
            theta_para=generate_theta_para(K,J,P_para,Z_para,n_count_para,Nuj_para,a,b,users)
            


            
            theta_convergenceinf[n]+=(1/M)*np.linalg.norm(theta-Truetheta,ord=2)*1/(K*J)
            alpha_convergenceinf[n]+=(1/M)*np.linalg.norm(alpha-Truealpha,ord=2)*1/(K)
            
            for k in range(K):
                trace_matrix_alpha[k,n]+=1/M*np.abs(alpha[k]-Truealpha[k])
                for j in range(J):
                    trace_matrix_theta[k,j,n]+=1/M*np.abs(theta[k][j]-Truetheta[k][j])
            
            Nuj_numpy=Nuj.to_numpy()
            


            for u in range(P):
                for k in range(K):
                    if Z[u]==k:
                        for j in range(J):
                                #value=df_sum[df_sum['users']==u+1]['J'+str(j+1)].tolist()[0]
                                value=df_sum_numpy[u][j+1]
                                if n==0:
                                  if value==-1:
                                    df_0_benchmark_numpy[u][j+1]=0
                                    df_1_benchmark_numpy[u][j+1]=1
                                    
                                    #df_0_benchmark.loc[Nuj['users']==u+1,'J'+str(j+1)]=0
                                    #df_1_benchmark.loc[Nuj['users']==u+1,'J'+str(j+1)]=1

                                  #if value==-2:
                                   # df_0_benchmark.loc[Nuj['users']==u+1,'J'+str(j+1)]=0
                                    #df_1_benchmark.loc[Nuj['users']==u+1,'J'+str(j+1)]=1

                                    
                                if value==-1:
                                  Nuj_numpy[u][j+1]=np.random.poisson(theta[k,j])
                                  #Nuj.loc[Nuj['users']==u+1,'J'+str(j+1)]=np.random.poisson(theta[k,j])
                

            
          
            for u in range(len(users)):
              
                for k in range(K):
                    if Z_para[u]==k:
                        for j in range(J):
                                #value=df_sum[df_sum['users']==u+1]['J'+str(j+1)].tolist()[0]
                                value=df_sum_numpy[users[u]][j+1]

                                if value==-1:
                                  Nuj_para.loc[Nuj_para['users']==users[u]+1,'J'+str(j+1)]=np.random.poisson(theta_para[k,j])                                           
                                
                                   
            
                                
        
            generate_Nuj(Nuj_numpy,theta,Z,K,J,P,S,Isum,df_0_benchmark_numpy,df_1_benchmark_numpy)

            Nuj=pd.DataFrame(Nuj_numpy,columns=['users']+['J'+str(j+1) for j in range(J)])
            df_0_benchmark=pd.DataFrame(df_0_benchmark_numpy,columns=['users']+['J'+str(j+1) for j in range(J)])
            df_1_benchmark=pd.DataFrame(df_1_benchmark_numpy,columns=['users']+['J'+str(j+1) for j in range(J)])

            #reach_true,reach_pred=calcul_reach(df_sans_independance,Nuj)
            reach_pred=0
            for u in range(P):
              if np.sum(Nuj_numpy[u][1:])>0:
                reach_pred+=1.0/P
            reach['Gibbs'][n]+=1/M*np.abs(reach_pred-reach_true)
            sd["Gibbs"][n,m]=np.abs(reach_pred-reach_true)

            

            
          
            

            aux,reach_bench0=calcul_reach(df_sans_independance,df_0_benchmark)
            aux,reach_bench1=calcul_reach(df_sans_independance,df_1_benchmark)
            reach['Bench'][n]+=1/M*np.abs(reach_bench0-reach_true) 
            reach['Bench1'][n]+=1/M*np.abs(reach_bench1-reach_true)

        Classes[m]=Z
        for n in range(N):
            res[n]=np.std(sd["Gibbs"][n])
        
        Nuj_para_numpy=Nuj_para.to_numpy()
        
        for u in range(len(Isum)):
          classe=np.random.rand()
          for k in range(K):
            
            if np.sum(alpha_para[:k]) <= classe < np.sum(alpha_para[:k+1]):
              Nuj_para_numpy=np.append(Nuj_para_numpy,np.array([Isum[u]]+[np.random.poisson(theta_para[k][j]) for j in range(J)]))
        Nuj_para_numpy=Nuj_para_numpy.reshape((P,J+1))
        
        reach_pred=0
        for u in range(P):
              if np.sum(Nuj_para_numpy[u][1:])>0:
                reach_pred+=1.0/P

        for n in range(N):
          reach['Para'][n]+=1/M*np.abs(reach_true-reach_pred)
    print('\n')
    print('\n')
    print('Gibbs Sampler')
    print(analyse_matrice(Nuj_numpy,df_sans_independance.to_numpy(),Z_true,K))
    print('\n')
    print('Estimation des paramètres')
    print(analyse_matrice(Nuj_para_numpy,df_sans_independance.to_numpy(),Z_true,K))
    print('\n')
    print('\n')

            
                
                                    #print(f'parametre theta={a+aux[k,j],1/(b+n_count[k])},theta={theta[k,j]} et Nuj={Nuj[Nuj["users"]==u+1]["J"+str(j+1)].tolist()[0]}')



    print('le theta estimé est {}'.format(theta))
    
    print('le vrai theta est {}'.format(Truetheta))

    
    
    #plt.figure()
    #plt.plot(Nitier,Number_of_right_answer['Bench'],label='Benchmark')
    #plt.plot(Nitier,Number_of_right_answer['Gibbs'],label='Gibbs')
    #plt.title('Proportion de bonnes prédictions')
    #plt.xlabel("Nombre d'itérations")
    #plt.ylabel('Proportion de bonnes réponses')
    #plt.legend()
    #plt.show()
    
    #plt.figure()
    #plt.plot(Nitier,ecart_df['Gibbs'], label='Gibbs')
    #plt.plot(Nitier,ecart_df['Bench'], label='Bench')
    #plt.title('Moyenne des Écarts avec df')
    #plt.legend()
    #plt.show()

    plt.figure()
    plt.plot(Nitier,NMI,label="Gibbs-Sampler" )
    plt.plot(Nitier,NMI_bench,label="Bench" )
    
    plt.title('Homogénéité des users entre les classes prédites et les vraies classes')
    plt.xlabel("Nombre d'itérations")
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    
    
    plt.figure()
    plt.plot(Nitier,theta_convergenceinf,label=r'$||\hat{\theta}-\theta||_2$')
    plt.legend()
    plt.title('Convergence moyenne vers les vrais paramètres theta')
    plt.show()

    plt.figure()
    plt.plot(Nitier,alpha_convergenceinf,label=r'$||\hat{\alpha}-\alpha||_2$')

    plt.legend()
    plt.title('Convergence moyenne vers les vrais paramètres alpha')
    plt.show()
              
    
    
    plt.figure()
    plt.hist([Classes[m] for m in range(M)],label = ['simulation : {}'.format(m) for m in range(M)])
    plt.title("Classes des données au cours des simulations")
    plt.legend()
    plt.show()

    
    #plt.figure(figsize=(12,8))
    #plt.hist([Val_Prises[n] for n in range(N)],label = ['itérations : {}'.format(n) for n in range(N)])
    #plt.title("Valeurs Moyennes Prises par les Nuj au cours des itérations")
    #plt.legend()
    #plt.show()
    
    
    
    
    plt.figure()
    for k in range(K):
            plt.plot(Nitier,trace_matrix_alpha[k],label='k={}'.format(k))
    plt.title(r'Trace des $q_k$')
    plt.legend()
    plt.show()

   
    plt.figure()
    for k in range(K):
        for j in range(J):
            plt.plot(Nitier,trace_matrix_theta[k,j,:],label='k={}'.format(k) + 'et j={}'.format(j),linewidth=5)
    plt.title(r'Trace des $\theta_{k,j}$')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(12,9))                         
    plt.plot([i+1 for i in range(N)],reach['Para'],"-",label='Gibbs-Estimation des Paramètres',linewidth=3,color='green')
    plt.plot([i+1 for i in range(N)],reach['Gibbs'],"-",label='Gibbs',linewidth=3)
    plt.plot([i+1 for i in range(N)],np.array(reach['Gibbs'])+np.array(res),"1--",label='Écart type',linewidth=2,color='blue')
    plt.plot([i+1 for i in range(N)],np.array(reach['Gibbs'])-np.array(res),"1--",label='Ecart type',linewidth=2,color='blue')
    #plt.plot([i+1 for i in range(N)],reach['Bench'],"o--",label='Bench 0 ',linewidth=3,color='red')
    plt.plot([i+1 for i in range(N)],reach['Bench1'],"o--",label='Bench 1',linewidth=3,color='orange')


    

    plt.title('|REACH PRED-REACH TRUE|')
    plt.legend()
    plt.show()




    
    

    
    return Nuj_para_numpy
                            
                
            
        
