### Versão em Português

# Projeto de Télédétection Approfondie - M2 SIGMA

**Análise de Dinâmica de Vegetação e Caracterização de Landes**

Este repositório contém o trabalho final desenvolvido para o módulo de Sensoriamento Remoto Avançado (Télédétection Approfondie) do Master 2 SIGMA (ENSAT/Agrogéomatique), referente ao ano letivo 2025-2026.

##  Resumo do Projeto

O objetivo principal é estudar a dinâmica das estratas vegetais, com foco específico na caracterização de **landes** (família *Ericaceae*) na região dos **Pirineus (Série 23-24)**. O projeto utiliza séries temporais de imagens Sentinel-2 para diferenciar classes de ocupação do solo através de índices fenológicos e aprendizado de máquina.

##  Metodologia e Objetivos

A cadeia de processamento abrange:

1. **Análise Estatística:** Avaliação da distribuição de amostras (Sol Nu, Herbe, Landes, Arbre).
2. **Estudo Fenológico:** Utilização do índice **NARI** (*Normalized Anthocyanin Reflectance Index*) para detecção de antocianinas, marcador chave das landes.
3. **Classificação Supervisionada:** Implementação do algoritmo **Random Forest** com otimização via *GridSearchCV*.
4. **Validação:** Análise de precisão através de matrizes de confusão e métricas de qualidade.

##  Estrutura do Depósito (Dépot Git)

Seguindo as diretrizes do curso, este repositório armazena apenas o essencial para a reprodutibilidade:

* `my_function.py`: Script Python com as funções de suporte.
* `projet_telea_GAMADELIMA_Thiago.ipynb`: Notebook principal com todo o fluxo de trabalho e relatórios.
* `results/`: Pasta contendo as figuras e gráficos gerados para análise.

##  Ambiente de Trabalho

* **Plataforma:** Onyxia (SSPCloud) via VSCode Python.
* **Dados:** Séries temporais Sentinel-2 e amostras de verdade de campo (EPSG:32630).

---
Link: https://mlang.frama.io/cours-marc-lang/stable/sigmaM2_telea/sigmaM2_projet_landes.html
### Version en Français 

# Projet de Télédétection Approfondie - M2 SIGMA

**Analyse de la dynamique de la végétation et caractérisation des Landes**

Ce dépôt contient le travail final développé pour le module de Télédétection Approfondie du Master 2 SIGMA (ENSAT/Agrogéomatique), pour l'année universitaire 2025-2026.

## 📝 Résumé du Projet

L'objectif principal est d'étudier la dynamique des strates végétales, et plus particulièrement de caractériser les **landes** (famille des *Éricacées*) sur le site des **Pyrénées (Série 23-24)**. Le projet s'appuie sur des séries temporelles d'images Sentinel-2 pour différencier les classes d'occupation du sol via des indices phénologiques et l'apprentissage automatique.

##  Méthodologie et Objectifs

La chaîne de traitement comprend :

1. **Analyse Statistique :** Évaluation de la distribution des échantillons (Sol Nu, Herbe, Landes, Arbre).
2. **Étude Phénologique :** Utilisation de l'indice **NARI** (*Normalized Anthocyanin Reflectance Index*) pour détecter la présence d'anthocyanes.
3. **Classification Supervisée :** Entraînement d'un modèle **Random Forest** avec optimisation des hyperparamètres (GridSearchCV).
4. **Évaluation de la Qualité :** Analyse de la précision globale et des matrices de confusion.

##  Structure du Dépôt

Conformément aux consignes, ce dépôt contient uniquement les éléments nécessaires à la reproduction des résultats :

* `my_function.py` : Script Python contenant les fonctions outils.
* `projet_telea_GAMADELIMA_Thiago.ipynb` : Le notebook principal (rapport et code).
* `results/` : Dossier contenant les figures et graphiques produits.

##  Environnement de Travail

* **Plateforme :** Onyxia (SSPCloud) via un service VSCode Python.
* **Données :** Séries temporelles Sentinel-2 et vérité terrain (EPSG : 32630).
---
Link: https://mlang.frama.io/cours-marc-lang/stable/sigmaM2_telea/sigmaM2_projet_landes.html


