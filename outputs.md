# Project Outputs – Optimization of E-Commerce Product Recommendation System

This file summarizes all outputs and results obtained after implementing different recommendation techniques.  

---

## 1. Data Preprocessing
- Missing values were identified and handled.  
- Distribution of missing values was visualized.  
 
<img width="737" height="181" alt="image" src="https://github.com/user-attachments/assets/b1e977c6-1874-42ff-8821-297e4955d32b" />


<img width="523" height="285" alt="image" src="https://github.com/user-attachments/assets/b32df328-d96c-4338-85b6-3e8e7c166eb5" />

---

## 2. Content-Based Filtering
- Recommendations generated using TF-IDF and cosine similarity.  
- Personalized product suggestions based on descriptions and tags.  

<img width="504" height="144" alt="image" src="https://github.com/user-attachments/assets/d37b8376-6a97-41b4-8808-21a9924dbb9b" />

---

## 3. Collaborative Filtering
- Recommendations based on user–item interactions.  
- Captures hidden similarities between users and products.  

<img width="382" height="257" alt="image" src="https://github.com/user-attachments/assets/bdf8c0a6-811a-49bd-a8b0-a73a8739b3b9" />


---

## 4. Popularity-Based Filtering
- Recommendations ranked by mean ratings and number of ratings.  
- Useful for cold-start users.  

<img width="311" height="316" alt="image" src="https://github.com/user-attachments/assets/34d3c590-6705-4390-abc1-ec0d6be449e2" />

---

## 5. Hybrid Filtering (Content + Collaborative)
- Combined content similarity and collaborative signals.  
- Improved F1-scores compared to individual models.  

<img width="396" height="248" alt="image" src="https://github.com/user-attachments/assets/abf874f2-b780-419b-8c8a-c9208299a887" />


---

## 6. Hybrid + Popularity Filtering
- Combines hybrid scores with popularity metrics.  
- Best performing model overall (high recall + balanced accuracy).  

<img width="530" height="306" alt="image" src="https://github.com/user-attachments/assets/76b4cc86-2456-4893-9235-b6fe8ffb4e5a" />
  

---

## 7. Performance Analysis
- Precision, Recall, and F1-scores compared across models.  
- Visualizations showed Hybrid + Popularity performed best.  

<img width="713" height="560" alt="image" src="https://github.com/user-attachments/assets/77fb955d-06af-4267-8e3e-626907becca5" />

<img width="690" height="249" alt="image" src="https://github.com/user-attachments/assets/a4a48b2d-d738-45c5-a63f-9ffc123cf357" />


---

## Summary
- **Content-Based** → Good personalization, limited diversity.  
- **Collaborative** → Strong for active users, weak for cold-start.  
- **Popularity-Based** → Simple, solves cold-start, but not personalized.  
- **Hybrid** → Balanced approach, better accuracy.  
- **Hybrid + Popularity** → Best overall, combining personalization and trend awareness.  
