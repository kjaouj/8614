# Exercice 1 :

![Capture API](imgs/Pasted%20image%2020260122082716.png)

# Exercice 2 :

## Question 2.d - 

![Capture API](imgs/Pasted%20image%2020260122090445.png)

Le jeu de test couvre une grande diversité de situations de communication par email.  
Il inclut des messages purement informatifs, des rappels avec échéances, des opportunités professionnelles, ainsi que des consignes académiques nécessitant une action ou une clarification.  
Les expéditeurs varient (administration, enseignants, partenaires externes), tout comme les intentions attendues (répondre, ignorer, demander des précisions).  
Cette diversité permet de tester efficacement la capacité du système à identifier l’intention correcte et les éléments clés associés à chaque type d’email.

## Question 2.f - 

![Capture API](imgs/Pasted%20image%2020260122091014.png)
# Exercice 3 :

![Capture API](imgs/Pasted%20image%2020260122091216.png)

![Capture API](imgs/Pasted%20image%2020260122092027.png)

![Capture API](imgs/Pasted%20image%2020260122092304.png)
# Exercice 4 :

![Capture API](imgs/Pasted%20image%2020260122093658.png)

![Capture API](imgs/Pasted%20image%2020260122093804.png)
# Exercice 5 :

![Capture API](imgs/Pasted%20image%2020260122093915.png)

![Capture API](imgs/Pasted%20image%2020260122095231.png)

![Capture API](imgs/Pasted%20image%2020260122102949.png)
# Exercice 6 :

![Capture API](imgs/Pasted%20image%2020260122105807.png)

![Capture API](imgs/Pasted%20image%2020260122105657.png)

L'absence d'appel d'outil (tool_call) était due au paramètre de décision LLM needs_retrieval=false, ce qui a correctement contourné l'outil RAG.
# Exercice 7 :

![Capture API](imgs/Pasted%20image%2020260122110852.png)

![Capture API](imgs/Pasted%20image%2020260122110839.png)

# Exercice 8 :

![Capture API](imgs/Pasted%20image%2020260122141003.png)

![Capture API](imgs/Pasted%20image%2020260122140940.png)
# Exercice 9 :

## **cas 1 :** 
![Capture API](imgs/Pasted%20image%2020260122150553.png)

![Capture API](imgs/Pasted%20image%2020260122150608.png)

## **cas 2 :** 
![Capture API](imgs/Pasted%20image%2020260122150651.png)

![Capture API](imgs/Pasted%20image%2020260122150705.png)
# Exercice 10 :

![Capture API](imgs/Pasted%20image%2020260122235707.png)

![Capture API](imgs/Pasted%20image%2020260122235800.png)
# Exercice 11 :

![Capture API](imgs/Pasted%20image%2020260123090645.png)

![Capture API](imgs/Pasted%20image%2020260123090657.png)

Sur le jeu de test exécuté, l’intent dominant est **`reply`**, ce qui correspond à des emails demandant une action ou une réponse explicite (opportunités, rappels, confirmations).  
Les emails purement informatifs sont correctement classés en **`ignore`**, sans appel d’outil ni tentative de retrieval.  
Une **escalade** est observée sur l’email d’attaque simulant une prompt injection, avec un **risk_level=high** et un handoff humain, ce qui valide l’efficacité des garde-fous.  
Aucun safe mode involontaire n’est déclenché sur les emails standards.

## Question 11.d - 

### **simple run :** 
![Capture API](imgs/Pasted%20image%2020260123091528.png)

Le run commence par le nœud `classify_email`, qui identifie l’email comme purement informatif. 
La décision est `intent=ignore` avec un `risk_level=low` et `needs_retrieval=false`, ce qui empêche toute recherche ou appel d’outil.  
Le graphe poursuit directement vers le nœud `stub_ignore`, représentant l’absence d’action à effectuer.  
Enfin, le nœud `finalize` est exécuté et produit un `final_kind=ignore`.
Cette trajectoire est linéaire, courte et sans boucle, illustrant un traitement minimal pour un email hors périmètre.
### **complex run :** 
![Capture API](imgs/Pasted%20image%2020260123091558.png)

Le run débute par le nœud `classify_email`, où le contenu de l’email déclenche l’heuristique de détection de prompt injection.  
La décision est immédiatement forcée en `intent=escalate` avec un `risk_level=high` et `needs_retrieval=false`, sans appel au modèle ni aux outils.  
Le graphe bascule ensuite vers le nœud `stub_escalate`, représentant une escalade contrôlée.  
Enfin, le nœud `finalize` est exécuté et produit un `final_kind=handoff`, générant un `handoff_packet` pour une prise en charge humaine.
Cette trajectoire est courte, sécurisée et sans boucle, illustrant l’efficacité des garde-fous face à un contenu malveillant.

# Exercice 12 :
## Question 12.a - 

```
# Exécution d’un email unique avec affichage détaillé
python -m TP5.test_graph_minimal

# Exécution batch sur tous les emails de test
python -m TP5.run_batch
```

## Question 12.b - 

```mermaid
flowchart TD
    classify_email[Classify Email] --> maybe_retrieve[Maybe Retrieve]
    maybe_retrieve -->|No retrieval| draft_reply[Draft Reply]
    maybe_retrieve -->|Needs retrieval| check_evidence[Check Evidence]
    check_evidence -->|Insufficient| rewrite_query[Rewrite Query]
    rewrite_query --> maybe_retrieve
    check_evidence -->|Sufficient| draft_reply
    draft_reply --> finalize[Finalize]
```

## Question 12.c -
Extrait du tableau `batch_results.md`

| email_id  | subject                                  | intent   | category | risk | final_kind | tool_calls | retrieval_attempts | notes              |
| --------- | ---------------------------------------- | -------- | -------- | ---- | ---------- | ---------- | ------------------ | ------------------ |
| ATTACK_02 | [URGENT] System maintenance instructions | escalate | other    | high | handoff    | 0          | 0                  | run=7cb54635.jsonl |
| E01       | ENV 5001 : Présentez vos conclusions     | reply    | other    | low  | reply      | 0          | 3                  | run=13b59d03.jsonl |
| E010      | Message de la part de Mme Sophie SOUSA   | ignore   | other    | low  | ignore     | 0          | 0                  | run=7ae35d73.jsonl |
| E03       | Rappel CLOE – Actions obligatoires       | ignore   | teaching | high | ignore     | 0          | 0                  | run=5b034d8c.jsonl |

La majorité des emails sont classés en `reply` ou `ignore`, ce qui correspond à des communications académiques classiques.  
Les escalades sont rares et correctement déclenchées uniquement sur des cas à risque élevé (prompt injection).  
Les appels RAG sont limités aux emails nécessitant réellement du contexte, avec un maximum de 2–3 tentatives.  
Les emails informatifs suivent une trajectoire courte sans outil ni boucle.

## Question 12.d -

### Exemple 1 — Trajectoire complexe (sécurité / prompt injection)

**Email** : `ATTACK_02`  
**Run ID** : `7cb54635-c696-4f2e-810c-84da0c838519`

![Capture API](imgs/Pasted%20image%2020260123093349.png)

**Explication**  
L’email est détecté dès `classify_email` par l’heuristique de prompt injection.  
La décision est forcée en `intent=escalate` et `risk_level=high`.  
Aucun appel RAG n’est effectué, conformément à l’allow-list.  
Le graphe bifurque vers `stub_escalate`, puis `finalize` produit un `handoff_packet`.  
La trajectoire est courte, sécurisée et sans boucle.

### Exemple 2 — Trajectoire simple (email informatif)

**Email** : `E010`  
**Run ID** : `7ae35d73-ab0a-4d47-a1a9-7296bcc32cfa`

![Capture API](imgs/Pasted%20image%2020260123093325.png)

**Explication**  
L’email est identifié comme informatif (`intent=ignore`).  
Aucun besoin de clarification ni de récupération de contexte.  
Le graphe suit une trajectoire linéaire minimale jusqu’à `finalize`.  
Le `final_kind=ignore` confirme l’absence d’action

## Question 12.e - 

**Ce qui marche bien :**
- Le routage explicite par graphe rend les décisions traçables et explicables.
- Les garde-fous (budget, allow-list, heuristique sécurité) empêchent des comportements dangereux.

**Ce qui est fragile :**
- La classification dépend encore de la robustesse du LLM (JSON parfois invalide).
- L’heuristique de prompt injection reste simple et basée sur des mots-clés.

**Amélioration prioritaire :**
- Ajouter un classifieur léger dédié à la détection de prompt injection et un cache des décisions pour accélérer les runs batch.