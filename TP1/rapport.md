- **Étudiant** : KJAOUJ Aymane
- **Commands** : - Python -m venv 8614
                 - Source 8614/bin/activate
                 - pip install -r requirement 

# Exercice 2 :

## Question 2.a -

![Capture API](imgs/Pasted%20image%2020260108114721.png)

```Output
['Art', 'ificial', 'Ġintelligence', 'Ġis', 'Ġmet', 'amorph', 'osing', 'Ġthe', 'Ġworld', '!']
```

Dans le tokenizer GPT-2, le symbole spécial `Ġ` indique la présence d’un **espace avant le token**.  
GPT-2 ne gère pas les espaces comme des tokens séparés, mais les encode directement dans les tokens eux-mêmes.  
Ainsi, `Ġintelligence` signifie que le mot _intelligence_ est précédé d’un espace.

## Question 2.b -

![Capture API](imgs/Pasted%20image%2020260108115244.png)

| Token         | ID    | Remarque                   |
| ------------- | ----- | -------------------------- |
| Artificial    | 22302 | Mot fréquent, token unique |
| Ġintelligence | 11252 | Contient un espace initial |
| Ġmeta         | 4943  | Début d’un mot long        |
| morph         | 16505 | Fragment fréquent          |
| osing         | 1131  | Fin de mot                 |
| !             | 0     | Ponctuation isolée         |


Les **tokens** sont des unités textuelles (mots ou sous-mots) issues de la tokenisation.  
Les **token IDs** sont les identifiants numériques associés à chaque token dans le vocabulaire du modèle.
-> Le modèle de langage ne manipule que les IDs; les tokens sont une représentation lisible pour l’humain.
## Question 2.c -

Observations :
- Les mots courants comme _Artificial_ ou _world_ sont représentés par **un seul token**.
- Les mots longs ou moins fréquents (_metamorphosing_) sont découpés en **plusieurs sous-tokens** (`meta`, `morph`, `osing`).
- Les espaces sont intégrés directement dans les tokens via le préfixe `Ġ`.
- La ponctuation (`!`) est isolée en token indépendant.

Byte Pair Encoding favorise la réutilisation des fragments fréquents, ce qui permet une gestion efficace des mots inconnus sans faire exploser la taille du vocabulaire.
## Question 2.d -

![Capture API](imgs/Pasted%20image%2020260108123644.png)

Le mot _antidisestablishmentarianism_ est tokenisé séparément afin d’identifier précisément ses sous-unités. Le tokenizer GPT-2 le découpe en plusieurs sous-mots fréquents tels que _anti_, _establish_, _ment_ et _ism_. Cette segmentation illustre le principe du Byte Pair Encoding, qui permet de représenter efficacement des mots rares à partir de fragments réutilisables.

# Exercice 3 :

## Question 3.a -
![Capture API](imgs/Pasted%20image%2020260108124820.png)

La matrice des encodages positionnels a pour shape **(1024, 768)**.  
- La première dimension correspond au **nombre maximal de positions** que le modèle peut représenter (1024 tokens).  
- La seconde dimension correspond à la **dimension des embeddings**, identique à celle des embeddings de tokens (768).  
Chaque position dans la séquence est donc associée à un vecteur dense de dimension 768.

-> Le paramètre `n_positions` représente la **longueur maximale du contexte** que GPT-2 peut traiter : Dans un modèle de langage causal, cela signifie que le modèle ne peut conditionner sa prédiction que sur les **1024 tokens précédents** au maximum.  
Au-delà de cette limite, les positions ne sont plus représentables, ce qui impose un tronquage ou un découpage du texte.
## Question 3.b -
![Capture API](imgs/Pasted%20image%2020260108125418.png)

![Capture API](imgs/Pasted%20image%2020260108125508.png)

La visualisation PCA des positions 0 à 50 montre une **trajectoire continue** des points dans l’espace projeté.  
Les positions successives sont proches les unes des autres, indiquant une variation progressive des embeddings positionnels.  
Il n’y a pas de regroupements discrets, mais plutôt une structure lisse, suggérant une représentation ordonnée de la position.  
-> Cela indique que GPT-2 encode la position comme une information continue plutôt que catégorielle.
## Question 3.c -
![Capture API](imgs/Pasted%20image%2020260108130132.png)

![Capture API](imgs/Pasted%20image%2020260108130150.png)

- Lorsque l’on étend la visualisation aux positions 0 à 200, la trajectoire devient plus étendue et moins localement lisible.  
  Les positions restent organisées de manière continue, mais la densité de points augmente et certaines structures locales deviennent moins visibles.  
  Cela suggère que les encodages positionnels couvrent un espace plus large à mesure que la position augmente.  
  Cette propriété permet au modèle de distinguer efficacement des positions éloignées dans de longues séquences.
- Cette structure suggère que GPT-2 apprend une représentation progressive et distribuée des positions.  
  Les positions proches sont encodées par des vecteurs similaires, tandis que les positions éloignées occupent des régions différentes de l’espace latent. Cela permet au mécanisme d’attention de raisonner efficacement sur l’ordre et la distance relative entre tokens.  
  L’augmentation de l’échelle rend cependant la structure plus complexe, ce qui reflète la difficulté croissante à distinguer précisément des positions très éloignées.
# Exercice 4 :

## Question 4.a -
![Capture API](imgs/Pasted%20image%2020260108131110.png)

GPT-2 est un modèle de langage **causal** : à chaque position _t_, il prédit le token suivant à partir des tokens précédents. Ainsi, les logits à la position _t−1_ correspondent à la distribution de probabilité de _token_t_ conditionnellement aux tokens 0 à t−1.
-> C’est pourquoi la probabilité du token observé à l’index _t_ doit être lue dans les logits à l’index _t−1_.
## Question 4.b -
![Capture API](imgs/Pasted%20image%2020260108131704.png)

La perplexité mesure à quel point un modèle de langage est **surpris** par une séquence de tokens. Elle correspond à l’exponentielle de la moyenne des log-probabilités négatives par token.  
- une perplexité faible signifie que le modèle attribue une forte probabilité aux tokens observés.  
- une perplexité élevée indique que la phrase est peu probable selon le modèle.
## Question 4.c -
![Capture API](imgs/Pasted%20image%2020260108132114.png)

La phrase grammaticalement correcte présente une **perplexité relativement faible** (≈ 109), tandis que la phrase mal ordonnée atteint une **perplexité extrêmement élevée** (≈ 4596).  
Cela indique que GPT-2 attribue une probabilité globale **beaucoup plus faible** à la seconde séquence.

L’ordre des mots dans la phrase incorrecte viole fortement les régularités syntaxiques de l’anglais apprises par le modèle.  
Cette violation entraîne plusieurs tokens très improbables, ce qui provoque une **explosion de la perplexité**.  

-> Un tel écart (facteur ≈ 40) montre que GPT-2 est très sensible à la **grammaticalité et à la structure séquentielle** du langage.
## Question 4.d -
![Capture API](imgs/Pasted%20image%2020260108132942.png)
La phrase française présente une **perplexité nettement plus élevée** (≈ 383) que la phrase anglaise correcte (≈ 109), mais **beaucoup plus faible** que celle de la phrase anglaise mal ordonnée (≈ 4596).

Cela indique que GPT-2 reconnaît partiellement la structure de la phrase française, mais lui attribue une probabilité globale plus faible.  
Cette différence s’explique par le fait que GPT-2 est majoritairement entraîné sur des données **anglaises**, tandis que le français est moins représenté dans le corpus d’entraînement.

De plus, les mots français sont souvent **plus fragmentés en sous-tokens**, ce qui entraîne une accumulation de probabilités conditionnelles plus faibles et donc une perplexité plus élevée.  
Malgré cela, la phrase française reste grammaticalement cohérente, ce qui explique pourquoi sa perplexité demeure bien inférieure à celle de la phrase anglaise syntaxiquement incorrecte.

## Question 4.e -
![Capture API](imgs/Pasted%20image%2020260108133400.png)
Les propositions sont grammaticalement plausibles et commencent souvent par un **espace**, ce qui est cohérent avec le tokenizer GPT-2.  
On observe des continuations naturelles de la phrase, confirmant que le modèle capture des régularités sémantiques et syntaxiques.
# Exercice 5 :

## Question 5.a -
- **Seed utilisé :** `42`
Le seed est fixé afin de rendre les résultats **reproductibles**.  
Cela garantit que les opérations aléatoires (notamment lors du sampling) produisent les mêmes sorties d’une exécution à l’autre, ce qui est essentiel pour comparer objectivement les méthodes de génération.
## Question 5.b -
![Capture API](imgs/Pasted%20image%2020260108134926.png)
En greedy decoding, le modèle choisit systématiquement le token **le plus probable** à chaque étape.  
En relançant plusieurs fois, le texte généré est **strictement identique**, car aucun aléa n’est introduit.

**NB :**  GPT-2 a été entraîné sur **beaucoup d’articles de presse**, et puisque le Greedy decoding sélectionne le chemin le plus fréquent dans les données, il donne un résultat qui prend la forme d'un dialogue journalistique.
## Question 5.c -
Outputs examples:
```Seed 2
The future of artificial intelligence is not clear, but that could change. The early progress of AI has been largely due to the ability to do some things fairly quickly, like calculate things, but the future is not clear. The early progress of AI has
```

```Seed 3
The future of artificial intelligence is bright and bright. The future of the Internet of Things, and the future of the future of the Internet of Things industry.

The future of the Internet of Things, and the future of the Internet of Things industry
```

Le sampling introduit de la **stochasticité**, ce qui génère des textes variés d’une exécution à l’autre. Contrairement au greedy, les sorties sont moins déterministes et plus créatives.

Cette méthode est mieux adaptée à la génération créative qu’à l’optimisation stricte de la probabilité.

-> Rôle des paramètres
- **Température** : contrôle le degré de hasard (plus élevée → plus de diversité).
- **top-k** : limite le choix aux _k_ tokens les plus probables.
- **top-p** : choisit dynamiquement un ensemble de tokens cumulant une probabilité _p_.

## Question 5.d -
![Capture API](imgs/Pasted%20image%2020260108141628.png)

Dans cette expérience, l’ajout d’une pénalité de répétition élevée (`repetition_penalty = 2.0`) **n’a pas modifié la sortie générée** par rapport à la génération sans pénalité.  
Cela s’explique par le fait que, dans ce cas précis, le texte généré ne contient pas de **boucles de répétition immédiates**.

La pénalité de répétition agit principalement lorsque le modèle tend à réutiliser **exactement les mêmes tokens** plusieurs fois dans une courte fenêtre.  
Lorsque la génération est déjà fluide et peu redondante, la pénalité peut donc avoir **peu ou pas d’effet visible**.
## Question 5.e -
![Capture API](imgs/Pasted%20image%2020260108141941.png)
Avec une **température très basse (0.1)**, le modèle privilégie presque exclusivement les tokens les plus probables.  
Cela conduit à une génération **très conservatrice**, mais aussi à des **répétitions importantes**, comme la réutilisation quasi identique de la séquence _“The future of artificial intelligence is not.”_.  
Le modèle peine alors à sortir de formulations très fréquentes, ce qui peut dégrader la qualité globale du texte.

À l’inverse, une **température très élevée (2.0)** favorise fortement l’exploration de tokens peu probables.  
La génération devient beaucoup plus **diverse**, mais aussi **moins cohérente**, avec des associations surprenantes (Google, IBM, Watson, Stanford) et une structure parfois décousue.  
Cela illustre clairement le compromis entre **cohérence sémantique** (température basse) et **diversité / créativité** (température élevée).
## Question 5.f -
![Capture API](imgs/Pasted%20image%2020260108142625.png)

Avec le beam search (`num_beams = 5`), le modèle génère une phrase **complète, fluide et grammaticalement correcte** en une seule exécution.  
Le texte produit est **cohérent sémantiquement** et présente une structure bien formée, sans coupure abrupte.

Le contenu reste cependant **très générique**, utilisant des formulations larges et consensuelles.  
Cela s’explique par le fonctionnement du beam search, qui vise à maximiser la **probabilité globale de la séquence**, en explorant plusieurs continuations possibles avant de sélectionner la meilleure.
## Question 5.g -
![Capture API](imgs/Pasted%20image%2020260108143522.png)

L’augmentation du nombre de beams entraîne un **ralentissement du temps de génération**, bien que le texte produit reste inchangé.  
Cela s’explique par le fonctionnement du beam search, qui maintient simultanément plusieurs **séquences candidates** à chaque pas de génération.

À chaque nouveau token, le modèle doit :
- évaluer toutes les extensions possibles de chaque beam,
- conserver les `num_beams` meilleures séquences selon leur probabilité cumulée.

Ainsi, lorsque `num_beams` augmente, le nombre de calculs effectués à chaque étape croît également.  
Même si la séquence optimale finale est la même, le coût computationnel est plus élevé, ce qui explique le temps de génération plus long pour `num_beams = 20` que pour `num_beams = 10`.