# classification-knn-supervise
Implémentation d'un algorithme d'apprentissage supervisé (K-Nearest Neighbors) pour la classification automatisée de données

import numpy as np
import math
from scipy.spatial import distance as dist_scipy
from collections import Counter
from P01_utils import lire_donnees, visualiser_donnees, lire_donnees_numpy

# 3.2 Récupération et visualisation des données
X_train, y_train = lire_donnees(100)
X_test, y_test = lire_donnees(10)


# 3.3 Implémentation des 𝑘‑plus proches voisins en “pur Python”

def dist(X_i, X_j):
    if len(X_i) != len(X_j):
        raise ValueError ("vecteurs de meme longueurs")
    somme_Xy = 0
    for xi, xj in zip(X_i,X_j):
        somme_Xy += (xi-xj)**2
    return math.sqrt(somme_Xy)


#2)Implémentez une fonction qui permette d’obtenir les indices des 𝑘 plus proches voisin d’un individu
#de test parmi le jeu d’entraînement

def indices_k_plus_proches(X_train, X_test_i, k):
    if not X_train:
        raise ValueError("Le jeu d'entraînement X_train ne peut pas être vide.")
    
    if not isinstance(k, int) or k <= 0 or k > len(X_train):
        raise ValueError(f"k doit être un entier positif <= {len(X_train)}.")
    
    # Vérifier les dimensions : tous les points de X_train doivent avoir la même dim que X_test_i
    dim = len(X_test_i)
    for x in X_train:
        if len(x) != dim:
            raise ValueError("Tous les vecteurs doivent avoir la même dimension.")
    
    # Calcul des distances avec une boucle pour que le code soit pls claire
    distances = []
    for x in X_train:
        distance = dist(X_test_i, x)  # Assumes 'dist' function is defined elsewhere
        distances.append(distance)
    
    # On va trier les indices par distance croissante
    indices_tries = np.argsort(distances)
    
    # Retourner les k premiers
    return indices_tries[:k]



def classe_majoritaire(classes):
    """Calcule la classe la plus représentée dans une liste de classes."""
    c = Counter(classes)
    return c.most_common(1)[0][0]


#--3)calcule la classe la plus représentée dans la liste (ici "F").

def k_plus_proches_voisins_liste(X_train, y_train, X_test, k=1):
    """Prédit les classes pour le jeu de test en utilisant k-plus proches voisins (version liste)."""
    clas_pred = []
    for x_test in X_test:
        indices = indices_k_plus_proches(X_train, x_test, k)
        clas_voisins = [y_train[i] for i in indices]
        pred = classe_majoritaire(clas_voisins)
        clas_pred.append(pred)
    return clas_pred

# 3.4 Ré‑implémentation des 𝑘‑plus proches voisins en utilisant numpy

# Rechargez les données au format numpy
X_train_np, y_train_np = lire_donnees_numpy(100)
X_test_np, y_test_np = lire_donnees_numpy(10)

def k_plus_proches_voisins_numpy(X_train, y_train, X_test, k=1):
    """Prédit les classes pour le jeu de test en utilisant k-plus proches voisins (version numpy)."""
    distances = dist_scipy.cdist(X_test, X_train)
    indices = np.argsort(distances, axis=1)[:, :k]
    classes_voisins = y_train[indices]
    num_F = np.sum(classes_voisins == "F", axis=1)
    num_H = k - num_F
    predictions = np.where(num_F > num_H, "F", "H")
    return predictions.tolist()  # Retourne une liste pour correspondre à la version précédente


  
#print(k_plus_proches_voisins_numpy(X_train_np,y_train_np, k=1)) 

#Programme principal

#3.2
#print(visualiser_donnees(X_train, y_train, X_test))

indices = indices_k_plus_proches(X_train, X_test[0], k=2)
classes = [y_train[i] for i in indices]
print(classe_majoritaire(classes))  # On trouve la classe F

#Test de k_plus_proches_voisins_liste :
print(k_plus_proches_voisins_liste(X_train, y_train, X_test, k=3))  

#Test de k_plus_proches_voisins_numpy :
print(k_plus_proches_voisins_numpy(X_train_np, y_train_np, X_test_np, k=1)) 
print(indices_k_plus_proches(X_train, X_test[0], k=2))
