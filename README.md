# 🎬 Hybrid Movie Recommendation System

A personalized movie recommender that leverages **Collaborative Filtering**, **Matrix Factorization (SVD & NMF)**, and **Content-Based Filtering** using movie genres. Built with Python, powered by `Surprise`, `scikit-learn`, and deployed using `Streamlit`.

##  Features

- 📊 **Exploratory Data Analysis (EDA)** on MovieLens dataset  
- 🤝 **Collaborative Filtering**:  
  - User-Based  
  - Item-Based  
- 🧠 **Matrix Factorization**:  
  - Singular Value Decomposition (SVD)  
  - Non-negative Matrix Factorization (NMF)  
- 🎭 **Content-Based Filtering** using TF-IDF on genres  
- 🔁 **Hybrid Recommender** combining collaborative & content-based strategies  
- 🖥️ **Interactive Web UI** built with Streamlit  
- 📸 Poster display with genre tags and predicted ratings  
- 🎯 Precision@K, RMSE, and MAE evaluation metrics  

## 📁 Dataset

**MovieLens Dataset**  
Includes user ratings, movie titles, genres, and timestamps.

- `movies.csv` — movie_id, movie_title, movie_genres, poster_url  
- `ratings.csv` — user_id, movie_id, user_rating, timestamp

> Dataset Source: [MovieLens]([https://grouplens.org/datasets/movielens/](https://www.kaggle.com/datasets/sriharshabsprasad/movielens-dataset-100k-ratings))

## 🛠️ Tools & Technologies Used

**Languages & Frameworks:**  
Python, Streamlit

**Libraries & Models:**  
- 📚 Pandas, NumPy, Scikit-learn  
- 🎯 Surprise (SVD, NMF, KNNBasic)  
- 🧾 TfidfVectorizer (for genre similarity)  
- 📈 Seaborn, Matplotlib  

**Version Control:**  
Git, GitHub

**Environment:**  
Jupyter Notebook, VS Code

## 📦 Installation

# Clone the repository
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt

▶️ Running the App
streamlit run app.py
Then, open the link provided by Streamlit in your browser.

🧪 Evaluation Metrics
Each model is evaluated using:
RMSE (Root Mean Square Error)
MAE (Mean Absolute Error)
Precision@K (for top-N recommendations)

📸 Screenshots

<img src="https://github.com/user-attachments/assets/befd8b74-f349-4467-9f1c-d34ca1ba3a81" alt="Personalized Recommendations" width="800"/>
<img src="https://github.com/user-attachments/assets/7c7f17ce-200a-4d29-b097-48329246643c" alt="Content-Based Recommendations" width="800"/>

🔄 Recommendation Logic
Collaborative Filtering: Based on user-user or item-item similarity from historical ratings.
Matrix Factorization: Learns latent features using SVD/NMF.
Content-Based: TF-IDF vectorization of genres and cosine similarity.
Hybrid: SVD predictions + fallback to content similarity if needed.

📌 Future Work
Add genre/year filters
Add further details to movies
Add user rating history visualization
Explore deep learning-based recommenders

🤝 Contributors
Indu M
Gopika Gokulanadh
Ardra Pradeepkumar
S. Rajalakshmi
