# Customer Churn Prediction Project

## Overview

This project focuses on developing a machine learning solution to predict customer churn. Churn refers to customers leaving a business or discontinuing its services. Identifying customers at risk of churning can help businesses take proactive measures to retain them, thereby increasing customer lifetime value and improving overall profitability.

## Business Problem

Customer retention is a critical factor for business growth, especially in highly competitive markets. Acquiring a new customer is significantly more expensive than retaining an existing one. For this reason, understanding and predicting customer churn is crucial. The primary objectives of this project include:

- Identifying key drivers behind customer churn.
- Building a predictive model to estimate churn likelihood.
- Providing actionable insights for targeted retention strategies.

### Impact on Business

The churn prediction model and accompanying analysis will:

1. **Increase Customer Retention**: By identifying high-risk customers, businesses can implement personalized retention campaigns, leading to improved customer satisfaction and loyalty.

2. **Reduce Costs**: Retaining customers is more cost-effective than acquiring new ones. By reducing churn, businesses can save on customer acquisition expenses.

3. **Enhance Decision-Making**: The insights generated from the analysis enable data-driven strategies, helping to focus efforts on the most impactful retention activities.

4. **Improve Revenue Streams**: Retained customers are more likely to make repeat purchases or upgrade to premium services, directly impacting revenue growth.

5. **Optimize Resource Allocation**: By focusing retention efforts on high-value customers or those most at risk, businesses can use their resources more efficiently.

## Key Features

1. **Data Exploration and Preprocessing**:

   - Cleaned and prepared the dataset by handling missing values, encoding categorical variables, and scaling numerical features.
   - Conducted exploratory data analysis (EDA) to identify patterns and correlations.

2. **Feature Engineering**:

   - Engineered new features such as tenure groups, customer segmentation, and interaction frequencies.
   - Selected the most relevant features based on importance metrics.

3. **Model Development**:

   - Built and evaluated multiple machine learning models, including Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier.
   - Used hyperparameter tuning (RandomizedSearchCV) to optimize model performance.

4. **Model Evaluation**:

   - Evaluated models based on metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
   - Selected the best-performing model for deployment.

5. **Insights and Visualizations**:

   - Generated feature importance charts to identify key factors influencing churn.
   - Created clear and interactive visualizations using Plotly to communicate findings.

## Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Plotly, Streamlit
- **Tools**: Jupyter Notebook, Streamlit for interactive app deployment

## Deployment

The final model is deployed using Streamlit, providing an interactive interface for stakeholders to explore churn predictions and insights. The application includes:

- A dashboard for data visualization and analysis.
- A prediction tool to identify churn likelihood for individual customers.



## Conclusion

This churn prediction project enables businesses to proactively address customer churn by identifying at-risk customers and implementing targeted retention strategies. By leveraging machine learning and data-driven insights, businesses can improve customer satisfaction, reduce operational costs, and increase revenue, thereby achieving sustainable growth.

