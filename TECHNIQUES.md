# My_Torch — Analyseur Tactique d'Échecs From Scratch

Ce projet implémente un réseau de neurones multicouche (MLP) développé **intégralement à la main**, sans dépendre de frameworks comme PyTorch, TensorFlow ou Keras. L'objectif : prendre en entrée une position d'échecs encodée en notation FEN et déterminer instantanément l'état de la partie (**Nothing**, **Check**, **Checkmate**) ainsi que la couleur du joueur concerné.

Ce document résume le fonctionnement global du système, les choix d'architecture, la justification des hyperparamètres et les enseignements tirés de sa conception.

## 1. Le Défi : Évaluer un Échiquier Sans Moteur de Calcul

Classifier l'état d'une position d'iciels échecs uniquement avec des calculs matrest un problème atypique :

* Un moteur d'échecs classique (type Stockfish) calcule un arbre de coups légaux. Pour savoir s'il y a échec et mat, il teste si le roi a une case de fuite, si une pièce peut s'interposer ou si l'attaquant peut être pris.

* Notre réseau de neurones, lui, n'a **aucun moteur de jeu intégré**. Il ne peut pas simuler le coup suivant. Il doit déduire la situation uniquement par **reconnaissance de motifs spatiaux** sur une image statique de l'échiquier.

La frontière entre un simple *Échec* et un *Échec et mat* dépend souvent d'un seul détail (un fou planqué à l'autre bout du plateau capable d'intercepter une diagonale). C'est ce qui rend la tâche particulièrement exigeante pour un MLP dense.

## 2. Représentation des Données (Input Pipeline)

### 2.1 Encodage One-Hot (768 caractéristiques)

Un échiquier compte 64 cases, et chaque case peut accueillir l'une des 12 pièces possibles (Pion, Cavalier, Fou, Tour, Dame, Roi en blanc ou en noir) ou être vide.

Plutôt que d'attribuer des valeurs arbitraires aux pièces (qui créeraient un ordre mathématique artificiel sans rapport avec le jeu), nous aplatissons l'échiquier sous forme d'un vecteur binaire One-Hot :

$$
64 \text{ cases} \times 12 \text{ types de pièces} = 768 \text{ entrées binaires}
$$

Chaque position devient un vecteur très creux (*sparse*) de dimension 768, où seules les cases occupées valent $1$.

### 2.2 Sémantique du Trait et Couleur

La chaîne FEN précise quel joueur a le trait (`w` pour les Blancs, `b` pour les Noirs). Plutôt que de forcer le réseau à réapprendre deux fois la même logique pour les deux camps, le modèle classifie l'état tactique structurel ($\{0: \text{Nothing}, 1: \text{Check}, 2: \text{Checkmate}\}$), et le parser associe le résultat à la couleur active pour restituer la prédiction finale (ex. *Black Checkmate*).

## 3. Architecture et Mathématiques du Réseau

Le modèle retenu repose sur un perceptron multicouche à propagation avant (*Feedforward Neural Network*) entièrement vectorisé.

```
Entrée (768) ──> Dense(512) ──> Dense(256) ──> Dense(128) ──> Dense(64) ──> Sortie (3)
                   [ReLU]         [ReLU]         [ReLU]        [ReLU]       [Softmax]

```

### 3.1 Initialisation de He (Kaiming Normal)

Avec des activations **ReLU**, la moitié des neurones s'éteint en moyenne à chaque couche. Une initialisation naïve mènerait rapidement à l'évanouissement ou à l'explosion du gradient.

Les matrices de poids sont donc initialisées selon la loi de He :

$$
W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}^{[l]}}}\right)
$$

Les biais $b^{[l]}$ débutent à zéro.

### 3.2 Propagation Avant (Forward Pass)

Pour chaque couche intermédiaire $l$ :

$$
Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}
$$

$$
A^{[l]} = \max(0, Z^{[l]})
$$

En sortie, les scores bruts (logits) sont transformés en probabilités avec une fonction Softmax stabilisée numériquement (soustraction du maximum) :

$$
\hat{Y}_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{3} e^{z_j - \max(z)}}
$$

### 3.3 Rétropropagation Analytique

La fonction de coût utilisée est la **Cross-Entropy catégorielle**, pondérée par une régularisation $L_2$ (*Weight Decay*) :

$$
J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{3} Y_{ik} \ln(\hat{Y}_{ik} + \epsilon) + \frac{\lambda}{2m} \sum_{l=1}^{L} \Vert{}W^{[l]}\Vert{}_F^2
$$

La combinaison Softmax + Cross-Entropy donne un gradient de sortie particulièrement propre pour démarrer la rétropropagation :

$$
\delta^{[\text{sortie}]} = \frac{1}{m} (\hat{Y} - Y)
$$

Les gradients remontent ensuite couche par couche via le produit de Hadamard avec la dérivée de ReLU ($\mathbb{I}(Z > 0)$).

## 4. Justification des Hyperparamètres (`configs/test_2.conf`)

Voici la configuration finale issue de nos expérimentations :

```
{
  "layers": [768, 512, 256, 128, 64, 3],
  "learning_rate": 0.05,
  "momentum": 0.9,
  "decay": 0.0001,
  "epochs": 100,
  "batch_size": 256,
  "dropout": 0.0,
  "print_every": 1,
  "device": "cuda"
}

```

### Pourquoi ces valeurs précises ?

* **Topologie pyramidale (`[768, 512, 256, 128, 64, 3]`)** :
  L'information doit être compressée progressivement. La couche à 512 neurones agrège les configurations brutes de cases. Les couches intermédiaires (256, 128) combinent les relations de pièces (lignes ouvertes, pièces protectrices). La couche à 64 neurones sert de goulot d'étranglement (*bottleneck*) pour synthétiser ces motifs avant les 3 neurones de décision.

* **Taux d'apprentissage (`learning_rate: 0.05`)** :
  Un pas à $0.01$ était trop lent pour converger efficacement dans un espace à plus de 600 000 paramètres. À l'inverse, dépasser $0.1$ entraînait des divergences brutales. La valeur $0.05$ s'est avérée le compromis idéal entre vitesse de descente et stabilité.

* **Momentum de Polyak (`momentum: 0.9`)** :
  La topologie de la fonction de perte pour ce type de données présente de nombreux plateaux. Le momentum conserve une fraction des directions précédentes ($V_t = 0.9 V_{t-1} + 0.1 \nabla J$), ce qui évite au gradient de stagner dans les zones plates et amortit les oscillations transversales.

* **Weight Decay (`decay: 0.0001`) vs Dropout (`0.0`)** :
  Aux échecs, masquer aléatoirement un neurone d'entrée ou de couche intermédiaire (principe du Dropout) revient conceptuellement à faire disparaître une pièce du plateau : un échec et mat peut instantanément devenir une position anodine. Le Dropout perturbait fortement l'apprentissage. La régularisation $L_2$ avec un léger coefficient ($\lambda = 10^{-4}$) s'est révélée bien plus efficace pour empêcher l'explosion des poids sans casser la cohérence du jeu.

* **Batch Size (`256`) et Accélération GPU (`device: "cuda"`)** :
  Traiter les positions par lots de 256 permet de saturer les cœurs de calcul GPU via CuPy. Cela apporte également juste assez de bruit stochastique pour empêcher le modèle de s'enfermer trop tôt dans un minimum local.

* **Nombre d'époques (`100`)** :
  À 100 époques, la courbe de perte sur le set de validation atteint son palier de stabilisation. Pousser plus loin n'apportait aucun gain significatif et risquait d'amorcer du surapprentissage.

## 5. L'Évolution du Projet : Ce Qu'on en Retient

En regardant l'ensemble du projet du premier commit jusqu'au modèle final, le développement a franchi quatre grandes étapes :

### 1. Du code mathématique pur au moteur vectorisé

Le projet a débuté par l'écriture des équations fondamentales en pur NumPy. L'enjeu initial était de vectoriser l'intégralité des opérations matricielles pour bannir toute boucle Python sur les exemples ou les neurones. Le calcul analytique du gradient a été systématiquement validé pour éliminer tout risque de dérive numérique.

### 2. Le passage à l'échelle (CPU vers GPU)

Entraîner des dizaines de milliers de positions sur un processeur classique prenait rapidement des heures. En intégrant un backend GPU via **CuPy** (`scripts/train_gpu.py`), les calculs de tenseurs ont été déportés sur la carte graphique, réduisant le temps d'entraînement par époque d'un facteur 15. Cette accélération a rendu possible l'itération rapide sur les hyperparamètres.

### 3. La rigueur sur les données et la symétrie

Pour enrichir le jeu de données, un script d'augmentation par symétrie verticale et inversion de couleur a été développé. Nous avons pu constater expérimentalement l'importance cruciale de la chaîne de traitement :

* L'augmentation doit impérativement être appliquée **uniquement sur l'ensemble d'entraînement**.

* Si des variantes miroirs se retrouvent réparties entre l'entraînement et la validation, le modèle mémorise les positions au lieu de généraliser, affichant un score trompeur.

* En isolant strictement la validation et le test avant toute altération, le réseau a atteint un score réel et reproductible entre **83 % et 86 %**.

### 4. La frontière théorique d'un MLP aux échecs

Ce projet illustre concrètement ce qu'un réseau dense peut et ne peut pas faire :

* Il excelle à repérer les positions saines (**Nothing** > 90 % de réussite) et les menaces directes (**Check** \~82 %).

* Ses erreurs résiduelles se concentrent sur la nuance fine entre *Check* et *Checkmate*. Sans exploration dynamique des coups possibles, le modèle prédit le mat par proximité spatiale (ex. roi acculé dans un coin face à des pièces lourdes), ce qui constitue la limite naturelle d'une analyse statique sans moteur de recherche.