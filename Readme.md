
# Au sujet du fichier csv : 

Ce dataframe contient les colonnes suivantes :

Input variables (based on physicochemical tests):
- fixed acidity | Acidité fixe : il s'agit de l'acidité naturelle du raisin comme l'acide malique ou l'acide tartrique.
- volatile acidity | Acidité volatile : l'acidité volatile d'un vin est constituée par la partie des acides gras comme l'acide acétique appartenant à la série des acides qui se trouvent dans le vin soit à l'état libre, soit à l'état salifié. L'acidité volatile donne au vin du bouquet.
- citric acid | Acide citrique : utilisé pour la prévention de la casse ferrique et participe au rééquilibrage de l'acidité des vins. 
- residual sugar | Sucre résiduel : sucres (glucose + fructose) encore présents dans le vin après fermentation.
- chlorides | Chlorures : matière minérale contenue naturellement dans le vin (sel, magnésium...)
- free sulfur dioxide | Sulfites libres : exhacerbent les propriétés antioxydantes du vin
- total sulfur dioxide | Sulfites libres + Sulfites liées à la réaction avec d'autres molécules du vin
- density | Densité du vin (g/l)
- pH | PH du vin
 - sulphates | Sulfates : sels composés d'anions SO4(2-) != sulfites
 - alcohol | degré d'alcool

Output variable (based on sensory data):
- quality | Qualité générale : note comprise en 0 et 10

# Au sujet du modèle 
Pour tester avec votre propre modèle entrainé, il faut qu'il s'appelle 'model.pkl'

# Choix effectués 
Pour réaliser le modèle nous avons choisi randomForest car cela donnait la meilleure précision avec nos données. 
Puis, on a choisi de séparer les données en 80% de train 20% de test.

Pour recupérer le vin parfait :  
on trie le tableau par ordre décroissant de qualité puis on fait une moyenne des valeurs pour estimer le meilleur vin 


# Contributeurs : 
 - Hugo Cambra Lefebvre : cambralefe@cy-tech.fr
 - Titouan Riot : riottitoua@cy-tech.fr
