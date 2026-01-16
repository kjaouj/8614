# Exercice 1 :

![Capture API](imgs/Pasted%20image%2020260116091045.png)

![Capture API](imgs/Pasted%20image%2020260116091104.png)![Capture API](imgs/Pasted%20image%2020260116091120.png)
# Exercice 2 :

![Capture API](imgs/Pasted%20image%2020260116092046.png)

![Capture API](imgs/Pasted%20image%2020260116093151.png)

![Capture API](imgs/Pasted%20image%2020260116093347.png)

![Capture API](imgs/Pasted%20image%2020260116093655.png)
# Exercice 3 :

![Capture API](imgs/Pasted%20image%2020260116094921.png)

![Capture API](imgs/Pasted%20image%2020260116095018.png)

# Exercice 4 :

![Capture API](imgs/Pasted%20image%2020260116095911.png)

![Capture API](imgs/Pasted%20image%2020260116095925.png)

**Question “Comment valider une UE ?”**  
 >Le retrieval est satisfaisant : les premiers résultats proviennent de documents administratifs officiels et contiennent directement la réponse. La légère redondance est acceptable et même bénéfique.

**Question “Sujets de PFE supplémentaires”**  
 >Le retrieval est insuffisant : les documents retournés ne contiennent pas la réponse dans le top-5. Cela s’explique par un chunking trop fin et un nombre de résultats insuffisant pour une question spécifique.  
 
 L'amélioration que j'ai tentée consiste à augmenter la taille des blocs et la valeur de TOP_K. Voici donc les nouvelles valeurs que j'ai essayées :
```
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 8
```

Voici donc les résultats :

![Capture API](imgs/Pasted%20image%2020260116105649.png)

![Capture API](imgs/Pasted%20image%2020260116105706.png)

Avec l’augmentation de la taille des chunks (`CHUNK_SIZE = 1200`, `CHUNK_OVERLAP = 200`) et du nombre de documents retournés (`TOP_K = 8`), le comportement du retrieval s’améliore de manière contrastée selon le type de question.

Pour la question **« Comment valider une UE ? »**, le retrieval est **très satisfaisant** : les premiers résultats proviennent quasi exclusivement de **documents administratifs officiels (PDF)** et contiennent directement les informations pertinentes (crédits ECTS, règles de validation, cas particuliers). Les chunks retournés sont cohérents et redondants de façon utile, ce qui confirme une bonne couverture du sujet.

En revanche, pour l'autre question, le retrieval reste **insuffisant** malgré les ajustements. Bien que davantage d’emails soient remontés, les premiers résultats ne contiennent toujours pas explicitement l’information recherchée. Cela suggère que les éléments clés (nom de la personne et sujets de PFE) sont soit peu présents dans le corpus, soit encore trop dispersés, ce qui limite la capacité du retriever à faire émerger la réponse.

# Exercice 5 :

![Capture API](imgs/Pasted%20image%2020260116111146.png)

![Capture API](imgs/Pasted%20image%2020260116111155.png)

![Capture API](imgs/Pasted%20image%2020260116111550.png)
# Exercice 6 :

![Capture API](imgs/Pasted%20image%2020260116115110.png)

![Capture API](imgs/Pasted%20image%2020260116113554.png)

![Capture API](imgs/Pasted%20image%2020260116113610.png)
**-> Score : 2**

![Capture API](imgs/Pasted%20image%2020260116113631.png)
**-> Score : 2**

![Capture API](imgs/Pasted%20image%2020260116113641.png)
**-> Score : 2**

## Question 6.g - 

![Capture API](imgs/Pasted%20image%2020260116114613.png)
La question est composite et nécessite des informations issues de plusieurs types de documents. Le retriever privilégie fortement les documents réglementaires (admin_pdf), ce qui conduit à une réponse correcte sur les règles académiques mais incomplète concernant les démarches administratives, souvent décrites dans les emails.

**Cas d’échec partiel – Question multi-sources**  
Pour une question combinant des règles académiques et des démarches administratives, le système produit une réponse correcte mais incomplète. Le retrieval privilégie fortement les documents réglementaires, ce qui permet de répondre précisément sur la validation des UE, mais ne couvre pas suffisamment les démarches administratives, souvent décrites dans les emails. Ce biais met en évidence une limite du pipeline RAG simple, qui pourrait être amélioré par un retrieval multi-filtres ou une stratégie adaptative selon le type de question.

**Conclusion**  
Le système RAG mis en place fonctionne correctement pour des questions factuelles et réglementaires, avec un retrieval cohérent et des réponses générées en français, sourcées et sans hallucination. La principale limite rencontrée concerne les questions nécessitant des informations issues de plusieurs types de documents, pour lesquelles le retrieval privilégie un type de source au détriment de l’autre, conduisant à des réponses partielles. En cas d’absence d’information, le système adopte un comportement sûr en indiquant explicitement une information insuffisante. L’amélioration prioritaire en vue d’un déploiement serait la mise en place d’une stratégie de retrieval adaptative ou multi-filtres, afin de mieux couvrir les questions transverses combinant règles administratives et échanges contextuels.