# Heart_Disease_Prediction

***Title: Predictive Modeling for Heart Disease Diagnosis using Machine Learning***

Heart disease remains a significant health concern globally. In an effort to aid in early detection and diagnosis, a machine learning project was undertaken using a dataset containing various clinical attributes related to heart health.

**Project Overview:**

The project aims to leverage machine learning algorithms to predict the likelihood of heart disease based on patient data. The dataset, sourced from "/content/heart.csv," comprises features such as age, cholesterol levels, blood pressure, and other medical indicators.

**Data Preprocessing:**

The initial step involved data loading and splitting into features (X) and the target variable (y). To evaluate the performance of multiple classifiers, the dataset was divided into training and testing sets (80/20 split) using train_test_split from the sklearn.model_selection module.

**Classifier Selection:**

**Several classifiers were employed to construct predictive models:**

Logistic Regression achieved an accuracy of approximately 78.54% with some variance (+/- 0.07).
Decision Tree Classifier and Random Forest Classifier both achieved an outstanding accuracy of 98.54%, showcasing excellent performance in predicting heart disease.
Gradient Boosting Classifier achieved an accuracy of around 93.17%, providing robust predictive capabilities.
AdaBoost Classifier achieved an accuracy of approximately 87.80%, demonstrating good performance in the classification task.
K-Nearest Neighbors (KNN) achieved an accuracy of around 73.17%, indicating moderate performance compared to other models.
Gaussian Naive Bayes achieved an accuracy of 80.00%, showing decent predictive capabilities.
Support Vector Classifier (SVC) using linear kernel achieved an accuracy of approximately 80.49%, similar to the Gaussian Naive Bayes.
Support Vector Classifier (SVC) using rbf kernel achieved an accuracy of around 68.29%, showing comparatively lower performance among the models.
Multilayer Perceptron (MLP) Classifier obtained an accuracy of 75.12%, which is relatively lower compared to most other models.
Additionally, a neural network model was trained using Keras (TensorFlow). 

The neural network achieved an impressive accuracy of **96.10%** on the test data, demonstrating its efficiency in predicting heart disease based on the provided features.

These results indicate that ensemble methods like Decision Trees, Random Forests, and Gradient Boosting perform exceptionally well in predicting heart disease, while the neural network achieved the highest accuracy among all models.

**Conclusion:**
Through this project, diverse machine learning algorithms were explored and assessed for their effectiveness in predicting heart disease based on patient attributes. The evaluation of various classifiers provides insights into the strengths and limitations of each model in diagnosing heart-related conditions.

In summary, leveraging machine learning models for predictive analytics in healthcare showcases promising avenues for early detection and improved patient care in the field of cardiology.
