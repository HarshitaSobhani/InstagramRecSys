Instagram Account Recommendation System

A hybrid recommendation system that suggests Instagram accounts using content-based filtering and collaborative filtering, combined into a single optimized engine. The project includes an interactive Streamlit dashboard for real-time recommendations and visual analysis.

Features:
Hybrid Recommendation Engine
Combines content-based and collaborative filtering
Adjustable blending factor (α)
Content-Based Filtering
TF-IDF vectorization of account descriptions
Cosine similarity for recommendation ranking
Collaborative Filtering
User-user similarity using cosine similarity
KNN-based neighbor aggregation (K=10)
Evaluation Metrics
Precision@K
Recall@K
Hit Rate (44% @ K=5)
Interactive Streamlit Dashboard
Select users dynamically
Adjust α and Top-N recommendations
Real-time visualizations
Data Visualization
Account-category heatmap
User-interest heatmap
Radial recommendation graph
Evaluation comparison charts

How It Works
1. Content-Based Filtering
Converts account descriptions into TF-IDF vectors
Builds a user profile from followed accounts
Recommends similar accounts using cosine similarity
2. Collaborative Filtering
Creates a user-account interaction matrix
Finds similar users using cosine similarity
Aggregates preferences from nearest neighbors
3. Hybrid Model

Final score is computed as:

Score = α × Content-Based + (1 - α) × Collaborative

Results
Hit Rate @5: 44%
Successfully recovers hidden user preferences in nearly half of the cases
Hybrid model improves recommendation quality over standalone methods


Project Structure

instagram-recsys/
│
├── dataset.py # Synthetic dataset generation
├── content_based.py # Content-based filtering
├── collaborative.py # Collaborative filtering
├── engine.py # Hybrid engine + evaluation
├── visualize.py # Visualizations
├── app.py # Streamlit dashboard
└── README.md

Installation & Setup
1. Clone Repository

git clone https://github.com/HarshitaSobhani/instagram-recsys.git
cd instagram-recsys

2. Install Dependencies

pip install -r requirements.txt

3. Run Application

streamlit run app.py

Example Output:
Personalized account recommendations
Dynamic heatmaps and graphs
Real-time tuning of hybrid model parameters

Use Cases:
Learning recommender systems
Machine learning portfolio project
Social media analytics experiments
Demonstrating hybrid recommendation techniques

Future Improvements:
Integration with real Instagram data
Deep learning-based recommendation models
Cold-start problem handling
Deployment on cloud platforms
🛠️ Tech Stack
Python
Scikit-learn
Pandas & NumPy
Streamlit
Matplotlib / Seaborn

Author:
Developed by Harshita Sobhani

⭐ Contribute

Feel free to fork this repository and improve it. Pull requests are welcome!
