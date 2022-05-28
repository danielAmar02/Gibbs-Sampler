# Mediametrie

Work realized for Mediametrie - French leader company which works on audience measurement - with Pr Guillaume Lecue.


In 2013, the World Federation of Advisers launched a call to effectively measure multi-media, multi-media and globally standardized advertising campaigns. Indeed, the current problem is that an advertising campaign usually takes place on several media (television, telephone, computer,...) and the measurement of advertising audience tends to be overestimated since an individual can be reached by the advertising campaign on several of these media.

However and very quickly, Google answered this call by proposing 3 papers [KSV13](https://research.google/pubs/pub41089/) [KSML16](https://research.google/pubs/pub45353/) [SK19](https://research.google/pubs/pub48387/). Not only the place of Google here invites reflection - Google being also an advertiser, it would become judge and jury - but above all the scientific approach of these papers raises questions [LP19](https://lecueguillaume.github.io/assets/XMM_stage.pdf). In particular, and as explained in this paper [LP19](https://lecueguillaume.github.io/assets/XMM_stage.pdf), the authors of [KSV13](https://research.google/pubs/pub41089/) [KSML16](https://research.google/pubs/pub45353/) [SK19](https://research.google/pubs/pub48387/) implicitly use the very unrealistic hypothesis that the number of times an individual is reached by an advertising campaign on one device is independent of the number of times he is reached by this same campaign on another device.


Here, we propose to explore a new approach using Bayesian statistics and the Gibbs-Sampler algorithm. This algorithm has many advantages for Médiamétrie because it allows to reconstitute a matrix that we will call the Cookie Matrix which counts for each user the number of times he has been reached on each device. 


| Users ID | Phone | Laptop | TV | $\dots$ | Radio | Total|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | $n_{1,p}$ | $n_{1,l}$ | $n_{1,t}$ | $\dots$ | $n_{1,r}$ | $n_1:=\sum_k n_{1,k}$|
| $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ | $\dots$ |
| P | $n_{p,p}$ | $n_{p,l}$ | $n_{p,t}$ | $\dots$ | $n_{p,r}$ | $n_p:=\sum_k n_{p,k}$|



Not only, this gives most accurate estimations of the reach - proportion of users reached by an advertising campaign - but it also allows to have more information.


# Simulation of the data 

In this project, we worked on simulated data using the Level-Correlated Model (LCM). The advantage of this model is that it allows to introduce some dependency among devices :
1. We assume that we have a population of size **P** subjected to an advertising campaign on **J** devices. Furthermore, we assume that the population can be partioned into K classes that represent, for example, socio-demographic classes. That is, each individual in the population belongs to a class $k \in [1,K]$. We will note $\alpha_k$ the proportion of individuals of class K.



