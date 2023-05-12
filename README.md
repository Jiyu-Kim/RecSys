<div align="center">

  <!--Header-->
  ![header](https://capsule-render.vercel.app/api?type=soft&color=auto&text=Recommender%20Systems)
  
</div>
<div>

  ## 📌 Introduction
  <!--This is a collection of assignments and projects that I completed during the <i><strong>Recommendation Systems</strong></i> course I took in the first semester of my fourth year.<br/><br/>-->
  This repository is a collection of lab practices where I reviewed research papers in the field of recommendation systems, identified state-of-the-art (SOTA) models, and implemented and validated those models mentioned in the papers during the <i><strong>Recommendation Systems</strong></i> course.<br/><br/>
  
  - HW1
    - Implement User-based collaborative filtering(models/userKNN_explicit.py)
    - Implement Item-based collaborative filtering(models/itemKNN_explicit.py)
    - naver_movie_dataset_small.csv and movielens_100k.csv are used to measure RMSE.
    <br/>
  - HW2
    - Implement **Slope One Predictors**(models/SlopeOnePredictor_explicit.py)
    - Implement **Matrix Factorization alogorithm with modeling user & item bias**(models/BiasedMF_explicit.py)
    - Implement **SVD++ algorithm**(models/SVDPP_explicit.py)
    - naver_movie_dataset_100k.csv and movielens_1m.csv are used to measure RMSE.
    <br/>
  - HW3
    - Implement **Implicit Matrix Factorization**(models/MF_implicit.py)
    - Implement **Weighted Matrix Factorization(WMF)**(models/WMF_implicit.py)
    - Implement **Bayesian Personalized Ranking(BPR) model**(models/BPR_implicit.py)
    - Implement **Factored Item Similarity Models(FISM)**(models/FISM_implicit.py)
    - Implement **Embarassingly Shallow Autoecnoders(EASE)**(models/EASE_implicit.py)
    - Implement **MRR function, NDCG funciton, and MAP function**(metrics.py)
    - naver_movie_dataset_100k.csv and movielens_1m.csv are used to measure MRR, NDCG and MAP.
    <br/>
  - HW4
    - Implement **Collaborative Denoising Autoencoder(CDAE) model**(models/CDAE_implicit.py)
    - Implement **Multinomial Variational Autoencoder(MultVAE) model**(models/MultVAE_implicit.py)
    - Implement **LightGCN model**(models/LightGCN_implicit.py)
    - naver_movie_dataset_100k.csv’ and ‘movielens_100k.csv are used to measure Precision, Recall and NDCG.
    <br/>
  - HW5
    - Implement **Field-Aware Factorization Model(FFM)**(models/FFM_explicit.py)
    - Implement **Deep Factorization Model(DeepFM)**(models/DeepFM_explicit.py)
    - naver_movie_dataset.csv is used to measure # of params over field dims.
  <br/>
  <br/>
  
  ## 🔧 Tech Stack
  ### Language
  <!--Python-->
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
  
  ### Framework
  <!--Pytorch-->
  <img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/>
  <br/>
  <br/>
  
  ## 💡 What I learned
  - Content-based recommender systems
  - Collablorative Filtering (CF)
  - Memory-based CF, Model-based CF
  - Neural/Deep CF
  - Context-enhanced CF, Factorization Machine (FM)
  - Sequential-aware RS, conversational RS
  - Cross-domain RS
  - Evaluation of Recommender Models (Precision and Recall at N, MAP, DCG, NDCG)
  
</div>
