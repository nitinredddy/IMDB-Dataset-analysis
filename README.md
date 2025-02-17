# Movie Review Sentiment Analysis

## Overview
This project is a sentiment analysis model for movie reviews using Natural Language Processing (NLP) and Machine Learning. The model classifies reviews as either positive or negative based on their text content. Two different models are implemented: Logistic Regression and LSTM (Long Short-Term Memory) neural network.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib & Seaborn (for data visualization)
- Scikit-learn (for Logistic Regression & data preprocessing)
- TensorFlow/Keras (for LSTM model)
- WordCloud (for text visualization)

## Dataset
The project uses the `IMDB Dataset.csv`, which contains movie reviews labeled as `positive` or `negative`.

## Features
- **Text Preprocessing:** Converts text to lowercase, removes special characters, and tokenizes.
- **TF-IDF Vectorization:** Transforms text data into numerical vectors for Logistic Regression.
- **Logistic Regression Model:** Trained to classify reviews as positive or negative.
- **LSTM Neural Network:** Trained for sentiment classification using word embeddings.
- **Data Visualizations:**
  - Sentiment distribution
  - Review length distribution
  - Confusion matrix for classification performance
  - Word cloud for common words in reviews

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow wordcloud
   ```
2. Place the `IMDB Dataset.csv` in the project directory.
3. Run the Python script to train models and visualize results:
   ```bash
   python sentiment_analysis.py
   ```

## Results
- Logistic Regression and LSTM models are trained and evaluated.
- Model accuracy and classification reports are displayed.
- Various visualizations help understand the dataset trends.

## Future Improvements
- Implement additional deep learning models (e.g., BERT)
- Optimize hyperparameters for better accuracy
- Deploy model as a web service using Flask or FastAPI


