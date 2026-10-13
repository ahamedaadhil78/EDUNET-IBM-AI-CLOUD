# Movie Recommendation System

## 📌 Project Overview

This project is a simple **Movie Recommendation System** built using Python and Machine Learning techniques.

The system recommends movies that are similar to a selected movie based on its **genre**.

It uses **TF-IDF Vectorization** and **Cosine Similarity** to calculate the similarity between movies.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

## ⚙️ How It Works

The recommendation system works in the following steps:

1. Movie data is created using Pandas.
2. Movie genres are converted into numerical features using **TF-IDF Vectorization**.
3. **Cosine Similarity** is calculated between all movies.
4. The system finds movies with the highest similarity score.
5. It returns the top 5 recommended movies.

## 🧠 Machine Learning Technique

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) converts the movie genre text into numerical values.

### Cosine Similarity

Cosine Similarity measures how similar two movie genre vectors are.

A higher similarity score means the movies have more similar genres.

## 📂 Project Structure

```text
Movie-Recommendation-System
│
├── Movie_Recommender_System.ipynb
└── README.md
