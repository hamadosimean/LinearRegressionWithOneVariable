#!/usr/bin/env python
# coding: utf-8

# # Problème: Mr DOE, un vendeur de poisson 🦈🦈🦈🦈🦈🦈veut savoir combien de poisson il va vendre aujourd'hui s'il regroupe plusieurs facteurs relatives à sa vente historique, par example le prix unitaire,le jour de pluie, location de la boutique, type de clientèle,etc
# 
# 
# # Objectif: Implementer un algorithm qui va permettre de faire la prediction de nombre de poisson de Mr DOE.
# 
# # Dans cette situation nous pouvons remarquer que la valeur à prédire est infinie( c'est à dire elle n'a pas de valeur finie, on peut avoir 400 poissons qui seront vendus, 500 qui seront vendus etc. Si c'etait finie on serait dans la situation, poisson carpe (1), autre type de poisson 2, un autre type 3 et j'en passe) . Par suite, nous allons implementer la regression linear qui permet de resoudre ce cas de problème
# 
# # Nous allons implement un algorithm qui permet d'apprendre la fonction fw,b(x)= w * x +b avec le vecteur x une variable et w,b les paramètres à rechercher. L'ojectif c'est de trouver w et b de telle sorte qu'ils soient des mininums globaux de f, alors l'algorithme de gradient descente sera utilisé.
# 
# # Note: Plusieurs paramètres devraient êtres prises en compte , mais comme j'avais mentionné en haut l'objectif c'est d'implementer l'algoithme et de voir comment ces algorithmes fonctionnent 
# 
# # Numpy est un module utilisé pour les operations mathematiques, algèbre linear , matplotlib c'est pour la visualisation (graphe)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import numpy as np
import matplotlib.pyplot as plt

class FatihamRegression():
    """Algorithme de prediction des valeur finies FatihamRegression
            parametres
            ----------
            n_iter : Nombre d'iteration
            learning_rate: Pas d'apprentissage
            random_state: pour l'initialisation des parametres
            verbose: pour voir certaines informations lors de
            l'appentissage, notamment l'evolution de la fonction du coût
    """
    def __init__(self, n_iter=2000, learning_rate=0.001,random_state=1,verbose=0):
        self.n_iter = n_iter # nombre d'iteration(pour la recherche de des parametres w et b)
        self.learning_rate = learning_rate #le pas d'apprentissage qui est entre [0,1]
        self.random_state = random_state #pour l'initialisation des parametres w,b
        self.verbose = verbose # Pour ajouter plus d'information a moment de l'apprentissage
    

    def cost(self,x,y,w,b):
        """Cette methode est la fonction du cout. Elle retourne la moyenne de lasomme des erreus de prediction"""
        cost = 0
        m = x.shape[0]
        for i in range(m):
            f = (w *x[i] + b)
            cost += (f - y[i])**2
        return (1/ (2 * m))*cost
    
    def compute_gradient(self,x,y,w,b):
        """Cette methode calcule la somme des derrivé partielles de la fonction et retourne la moyenne de ses sommes"""
        m = x.shape[0]
        d_w  = 0
        d_b = 0
        for i in range(m):
            fw = w *x[i] + b
            d_w += (fw-y[i]) * x[i]
            d_b += (fw-y[i])

        return d_w / m,d_b / m
    
    def compute_gradient_descent(self,x,y,w,b):
        """Cette methode fait la recherche de w et b. C'est le gradient descente algorithm. elle retourne w et b"""
        self.cost_history = []
        for i in range(self.n_iter):
            w_d,b_d = self.compute_gradient(x,y,w,b)
            w = w - self.learning_rate * w_d
            b = b - self.learning_rate * b_d 

            if self.n_iter <100000:
                self.cost_history.append(self.cost(x,y,w,b))
                if self.verbose ==1 or self.verbose == 2:
                    if i%10==0:
                        print(f"Iterration {i} cost {round(self.cost_history[i],5)}")

        return w,b


    
    def fit(self,x,y):
        """L'apprentissage ce fait à travers cette methode"""
        w_i = np.random.RandomState(self.random_state).normal(loc=0,scale=0.1,size=1)
        b_i = np.random.RandomState(self.random_state).normal(loc=0,scale=0.1,size=1)
        self.w,self.b = self.compute_gradient_descent(x,y,w_i,b_i)
        return self
    

    def predict(self,x):
        """Cette methode fait la prediction des valeurs"""
        return np.dot(x,self.w) + self.b


# # On fait prendre un example avec des nombres aleatoires. Je veux faire exprès de lui donnée les mêmes valeurs voir comment il va faire les predictions

# In[7]:


my_reg= FatihamRegression(n_iter=200000,learning_rate=0.001,verbose=1)

x = np.array([5,8,8,41,12,25])
y = np.array([5,8,8,41,12,25])

#Entrainer le modèle
my_reg.fit(x,y)


#Faire des predictions
print(my_reg.predict(np.array([10,7,9,8,6,9]).reshape(-1,1)))


# In[9]:


print(my_reg.predict(np.array([20])))


# # Comme on peut remarquer il a fait une prediction 10.0000706 et la vraie valeur est 10, 7.00008361 et la vraie valeur est 7, ainsi de suite. Allons celebrer ça 🍾🍾🍾🍾🍾🏃‍♀️🤾‍♀️🤾‍♀️🤸🤸🤸🤸🤸🤸🤸🤸‍♀️🤸‍♀️🤸‍♀️🤸‍♀️

# In[4]:


# On va voir est_ce que notre gradient descent converse vers le minimum global
plt.figure(figsize=(10,8))
plt.plot(my_reg.cost_history[:60],label="Gradient descent",linewidth=4)
plt.xlabel("Iteration",fontsize=14)
plt.ylabel("Cost",fontsize=14)
plt.title("Gradient descent",fontsize=14)
plt.legend(loc="best")
plt.show()


# # Conclusion Mr DOE peut desormais faire sa prediction de vente et bien d'autres choses avec cet algorithme 

# # Fonction

# In[5]:


plt.figure(figsize=(8,10))
plt.imshow(plt.imread("function.png"))
plt.show()


# # Gradient descent

# In[6]:



plt.figure(figsize=(8,10))
plt.imshow(plt.imread("gradient.png"))
plt.show()

