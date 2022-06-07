# Mediametrie

Work realized for Mediametrie - French leader company which works on audience measurement - with Pr Guillaume Lecue.


In 2013, the World Federation of Advertizers launched a call to effectively measure multi-media and globally standardized advertising campaigns. Indeed, the current problem is that an advertising campaign usually takes place on several media (television, telephone, computer,...) and the measurement of advertising audience tends to be overestimated since an individual can be reached by the advertising campaign on several of these media.

However and very quickly, Google answered this call by proposing 3 papers [KSV13](https://research.google/pubs/pub41089/) [KSML16](https://research.google/pubs/pub45353/) [SK19](https://research.google/pubs/pub48387/). Not only the place of Google here invites reflection - Google being also an advertiser, it would become judge and jury - but above all the scientific approach of these papers raises questions [LP19](https://lecueguillaume.github.io/assets/XMM_stage.pdf). In particular, and as explained in this paper [LP19](https://lecueguillaume.github.io/assets/XMM_stage.pdf), the authors of [KSV13](https://research.google/pubs/pub41089/) [KSML16](https://research.google/pubs/pub45353/) [SK19](https://research.google/pubs/pub48387/) implicitly use the very unrealistic hypothesis that the number of times an individual is reached by an advertising campaign on one device is independent of the number of times he is reached by this same campaign on another device.


Here, we propose to explore a new approach using Bayesian statistics and the Gibbs-Sampler algorithm. This algorithm has many advantages for Médiamétrie because it allows to reconstitute a matrix that we will call the Cookie Matrix which counts for each user the number of times he has been reached on each device using the information $c_j$ of the aggregate cookies per devices.


| Users ID | Phone | Laptop | TV | $\dots$ | Radio | Total|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $n_{1,p}$ | $n_{1,l}$ | $n_{1,t}$ | $\dots$ | $n_{1,r}$ | $n_1:=\sum_k n_{1,k}$|
| $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| P | $n_{p,p}$ | $n_{p,l}$ | $n_{p,t}$ | $\dots$ | $n_{p,r}$ | $n_p:=\sum_k n_{p,k}$|
| Total | $c_p=\sum_{i=1} n_{i,p}$ | $c_l=\sum_{i=1} n_{i,l}$ | $c_t=\sum_{i=1} n_{i,t}$  | $\dots$ | $c_r=\sum_{i=1} n_{i,r}$ | |




Not only, this gives most accurate estimations of the reach - proportion of users reached by an advertising campaign - but it also allows to have more information. We obtained 2 main results :
1. We were able to make an estimation of the reach without research panelist with 20% error on the reach accuracy. Research panelists are member of a panel  and agree to share their data to an audience measurement company. Usually, it is very difficult to recruit such panelists and they often disagree to give all of the data : _e.g_ they might say that they agree to be followed on their TV but not on their laptob. Thus, our estimated reach may not be a good estimation but it is already something new in comparison with google papers which use panelists.
2. With the use of panelists or bi-panelists (panelists who agree to be followed only on certain devices), we were able to make a significative improvement in comparison with Google papers.

# Simulation of the data 

In this project, we worked on simulated data using the Level-Correlated Model (LCM). The advantage of this model is that it allows to introduce some dependencies among devices :
1. We assume that we have a population of size **P** subjected to an advertising campaign on **J** devices. Furthermore, we assume that the population can be partioned into **K** classes that represent, for example, socio-demographic classes. That is, each individual in the population belongs to a class $k \in [1,K]$. We will note **$\alpha_k$** the proportion of individuals of class K.
2. To simulate the data, we start by uniformly assigning a class k to each user. We then obtain a vector **Z** of size P such that "Z[i]=k" means that the i-th user is of class k. We then obtain the true proportions of the classes in the population.
3. Then, we will create the cookie matrix: we first generate K positive semidefinite matrices $\Sigma_1, \dots, \Sigma_K$ of size J. We then generate $\gamma_{k1},\dots,\gamma_{kJ}$ according to $\mathcal{N}(0,\Sigma^k)$. 
4. Finally, if the individual u is of type k, we generate **$N_{uj}$** - the number of cookies received by u on the device j - by $\mathcal{P}(\theta_{kj}:=exp(\gamma_{kj}))$.
5. In order to have a more realistic reach, we decided to add a random (uniform) weight $\eta$ on a Dirac in 0 for all user u on class k in the device j : $N_{uj} \sim \eta_{kj}\delta_0 +(1-\eta_{kj})\mathcal{P}(\theta_{kj})$.

# A new approach using the Gibbs-Sampler : Without Panelist

This is the pseudo-code of the algorithm we used :

![alt text](https://github.com/danielAmar02/Mediametrie/blob/main/Media/algo_gs_woutpan.png)

Few remarks:
- We compared it with a benchmark that fills the cookie matrix with random 0's or 1's and with the Google model (by providing the Google the reach on the panelist)
- After few simulations, we obtained the results below. We see that we do not beat Google benchmark but let's recall that we provided more info to this model as Google can only proceed if it has information from panelists
- The Gibbs Sampler has an important complexity in $O(N_{tab}N_{iter}N_{test}PJM)$

![alt text](https://github.com/danielAmar02/Mediametrie/blob/main/Media/14.png)


# A new approach using the Gibbs-Sampler : With Panelist

![alt text](https://github.com/danielAmar02/Mediametrie/blob/main/Media/algo_gs_wpan.png)


We obtained the result below :


![alt text](https://github.com/danielAmar02/Mediametrie/blob/main/Media/R_P%253D300%252CMass0%253D0.85%252CK%253D2%252CJ%253D5%252CHP%253D4%252CM%253D4%252CN%253D15%252Cprop_bi%253D05.png)
