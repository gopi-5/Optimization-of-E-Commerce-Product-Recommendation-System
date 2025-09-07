# Project Outputs – Optimization of E-Commerce Product Recommendation System

This file summarizes all outputs and results obtained after implementing different recommendation techniques.  

---

## 1. Data Preprocessing
- Missing values were identified and handled.  
- Distribution of missing values was visualized.  

![Missing Values](images/missing_values.png)  
<img width="737" height="181" alt="image" src="https://github.com/user-attachments/assets/b1e977c6-1874-42ff-8821-297e4955d32b" />

![Percentage Missing](images/percentage_missing.png)  

---

## 2. Content-Based Filtering
- Recommendations generated using TF-IDF and cosine similarity.  
- Personalized product suggestions based on descriptions and tags.  

![Content-Based Recommendations](images/content_based.png)  

---

## 3. Collaborative Filtering
- Recommendations based on user–item interactions.  
- Captures hidden similarities between users and products.  

![Collaborative Recommendations](images/collaborative.png)  

---

## 4. Popularity-Based Filtering
- Recommendations ranked by mean ratings and number of ratings.  
- Useful for cold-start users.  

![Popularity Metrics](images/popularity_metrics.png)  
![Popularity Recommendations](images/popularity_recommendations.png)  

---

## 5. Hybrid Filtering (Content + Collaborative)
- Combined content similarity and collaborative signals.  
- Improved F1-scores compared to individual models.  

![Hybrid Recommendations](images/hybrid.png)  

---

## 6. Hybrid + Popularity Filtering
- Combines hybrid scores with popularity metrics.  
- Best performing model overall (high recall + balanced accuracy).  

![Hybrid+Popularity Recommendations](images/hybrid_popularity.png)  

---

## 7. Performance Analysis
- Precision, Recall, and F1-scores compared across models.  
- Visualizations showed Hybrid + Popularity performed best.  

![Performance Comparison](images/performance_comparison.png)  

---

## Summary
- **Content-Based** → Good personalization, limited diversity.  
- **Collaborative** → Strong for active users, weak for cold-start.  
- **Popularity-Based** → Simple, solves cold-start, but not personalized.  
- **Hybrid** → Balanced approach, better accuracy.  
- **Hybrid + Popularity** → Best overall, combining personalization and trend awareness.  
