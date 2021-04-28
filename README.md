# POC - Automation of text and images classification

Dataset: https://www.kaggle.com/PromptCloudHQ/flipkart-products

Context/Scenario: You are a Data Scientist at the company "Marketplace", which wants to launch an e-commerce marketplace. On the marketplace, sellers offer items to buyers by posting a photo and a description. The sellers need to manually enter products category. Some errors have been found in the website and the consumers take too many time to find the good product.

Problem: Define categories of images and description products with ML and DL

Method for text description:
1. Extraction of pertinent features
2. Lower case
3. Tokenizer
4. Lemmatizer
5. Delete stop words and 70 common words
6. Bag of words
7. Count vectorizer with different n-grams (bi-gram, tri-gram, combination of uni- and bi-gram, combination of uni-, bi- and tri-gram)
8. GridSearch CV for LDA hyperparameters
9. Topic modelling
10. Dimension reduction
11. Clustering

Method for images with SIFT:
1. Pre-processing with detect and compute
2. Descriptors and keypoints
3. KMeans application to obtain bag of visual words
4. Dimension reduction
5. Clustering

Method for images with VGG16:
1. Transform img to arrays
2. Preprocessing with Keras
3a. Train VGG16 with defined categories
3b. Train VGG16 with Transfer Learning
4. Confusion matrix
5. Plot tSNE reduction and clustering with images displayed in 2D

Results:
1. Best model in topic modelling: combination of uni- and bi-gram, tSNE and KMeans (Silhouette coef=0.8 and Davies Bouldin=0.31)
2. Transfer Learning results: Silhouette coef=0.39 and Davies Bouldin=0.81
3. Categories are well defined with 7 types of products in LDA output and VGG16 output
3. LDA Accuracy=0.93

Libraries: Pandas, Numpy, Matplotlib, Scikit-learn, NLTK, pyLDAvis, WordCloud, OpenCV,Keras
