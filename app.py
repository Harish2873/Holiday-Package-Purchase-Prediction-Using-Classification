import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title='Holiday Package Purchase Prediction',page_icon='🗺️',layout='centered',initial_sidebar_state='auto')

preprocessor = joblib.load("Models/preprocessor.pkl")
model = joblib.load('Models/gradientboosting_model.pkl')

page = st.sidebar.radio('Choose The Page',['Home','EDA','Model Comparison','Feature Importance','Prediction','About Project'])

if page == 'Home':
    st.title('🗺️ Holiday Package Purchase Prediction')

    st.header('🧩 Input Features')

    def user_input_features():
        Age = st.slider('Select the age',min_value=1,max_value=100,value=30)
        TypeofContact = st.selectbox('Type Of Contact ',['Self Enquiry', 'Company Invited'])
        CityTier = st.selectbox('City Tier',['Metro City (Tier 1)', 'Urban City (Tier 2)', 'Town/Village (Tier 3)'])
        DurationOfPitch = st.slider("Duration of Pitch (in mins)", 0, 120)
        Occupation = st.selectbox('Occupation',['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
        Gender = st.selectbox('Gender',['Male','Female'])
        NumberOfFollowups = st.selectbox('Number of Follow-ups', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ProductPitched = st.selectbox('Product Pitched',['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
        PreferredPropertyStar = st.radio('Preferred Property Star', [3.0, 4.0, 5.0])
        MaritalStatus = st.selectbox('Marital Status',['Married', 'Unmarried', 'Divorced'])
        NumberOfTrips = st.slider('Number of Trips per Year', 1, 20)
        Passport = st.selectbox('Passport',['Yes','No'])
        PitchSatisfactionScore = st.select_slider('Pitch Satisfaction Score', options=[1, 2, 3, 4, 5])
        OwnCar = st.selectbox('Own Car',['Yes','No'])
        Designation = st.selectbox('Designation',['Executive','Manager','Senior Manager','AVP','VP'])
        MonthlyIncome = st.number_input("Monthly Income", min_value=0)
        TotalVisits = st.slider("Total People Visiting", 1, 20)

        city_map = {'Metro City (Tier 1)': 1, 'Urban City (Tier 2)': 2, 'Town/Village (Tier 3)': 3}
        City = city_map[CityTier]

        Passport_val = 1 if Passport=='Yes' else 0
        OwnCar_val = 1 if OwnCar=='Yes' else 0

        input_dict = {
            'Age':Age,
            'TypeofContact' : TypeofContact,
            'CityTier' : City,
            'DurationOfPitch' : DurationOfPitch,
            'Occupation' : Occupation,
            'Gender' : Gender,
            'NumberOfFollowups' : NumberOfFollowups,
            'ProductPitched' : ProductPitched,
            'PreferredPropertyStar' : PreferredPropertyStar,
            'MaritalStatus' : MaritalStatus,
            'NumberOfTrips' : NumberOfTrips,
            'Passport' : Passport_val,
            'PitchSatisfactionScore' : PitchSatisfactionScore,
            'OwnCar' : OwnCar_val,
            'Designation' : Designation,
            'MonthlyIncome' : MonthlyIncome,
            'TotalVisits' : TotalVisits
        }

        return pd.DataFrame([input_dict])

    input_df = user_input_features()

    if st.button('Predict Product Taken'):
        X_Preprocessed = preprocessor.transform(input_df)
        Prediction = model.predict(X_Preprocessed)[0]
        
        if Prediction == 1:
            st.success('Product Taken')
        else:    
            st.success('Product Not Taken')
        
        input_df['ProductTaken'] = Prediction

        try:
            existing = pd.read_csv('Prediction_log.csv')
            new_log = pd.concat([existing, input_df],ignore_index=True)
        except FileNotFoundError:
            new_log = input_df
        
        new_log.to_csv('Prediction_log.csv',index=False)
        st.info('🔮 Prediction Logged.')

    if st.checkbox('🔮 Show Past Predictions'):
        try:
            log_df = pd.read_csv('Prediction_log.csv')
            st.dataframe(log_df)
        except FileNotFoundError:
            st.warning('🔮 No Predictions Made Yet.')


if page == 'EDA':
    st.header("📊 Exploratory Data Analysis")

    with st.expander("🔍 Show Raw Data"):
        df = pd.read_csv("Dataset/Travel.csv")
        st.dataframe(df)

    with st.expander("📈🔥 Correlation Heatmap"):
        corr = df.select_dtypes(include=['number']).corr()
        fig = plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(fig)

    with st.expander("📊 Target Distribution"):
        st.image("Images/Pie.png")


if page == 'Model Comparison':    
    st.header('Model Comparison')
    st.image('Images/Model-Comparison.png',caption='Performance Comparison')

    st.header('📈 ROC Curves of All Models')
    with st.expander('Logistic Regression ROC Curve'):
        st.image('Images/Roc-Auc-Curve/Log_roc_curve.png',caption='Logistic Regression ROC Curve')
    with st.expander('Support Vector Classifier ROC Curve'):
        st.image('Images/Roc-Auc-Curve/SVC_roc_curve.png',caption='Support Vector Classifier ROC Curve')
    with st.expander('K-Neighbors Classifier ROC Curve'):
        st.image('Images/Roc-Auc-Curve/K-Neighbors_roc_curve.png',caption='K-Neighbors Classifier ROC Curve')
    with st.expander('Gaussian Naive Bayes ROC Curve'):
        st.image('Images/Roc-Auc-Curve/GaussianNB_roc_curve.png',caption='Gaussian Naive Bayes ROC Curve')
    with st.expander('Decision Tree Classifier ROC Curve'):
        st.image('Images/Roc-Auc-Curve/DecisionTree_roc_curve.png',caption='Decision Tree Classifier ROC Curve')
    with st.expander('Random Forest Classifier ROC Curve'):
        st.image('Images/Roc-Auc-Curve/RandomForest_roc_curve.png',caption='Random Forest Classifier ROC Curve')
    with st.expander('AdaBoost Classifier ROC Curve'):
        st.image('Images/Roc-Auc-Curve/AdaBoost_roc_curve.png',caption='AdaBoost Classifier ROC Curve')
    with st.expander('Gradient Boosting Classifier ROC Curve'):
        st.image('Images/Roc-Auc-Curve/GradientBoosting_roc_curve.png',caption='Gradient Boosting Classifier ROC Curve')
    with st.expander('eXtreme Gradient Boost Classifier ROC Curve'):
        st.image('Images/Roc-Auc-Curve/XGB_roc_curve.png',caption='eXtreme Gradient Boost Classifier ROC Curve')


if page == 'Feature Importance':
    st.header('🏷️ Feature Importance (Gradient Boosting)')

    importances = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({'Feature':feature_names,'Importance':importances})
    fig2 = plt.figure(figsize=(10,6))
    sns.barplot(data=importance_df.head(15),x='Importance',y='Feature')
    st.pyplot(fig2)


if page == 'Prediction':
    st.subheader('Predicted Log Data')
    st.download_button('📥 Download Predicted Log',data=open('Prediction_log.csv').read(),file_name='Prediction_log.csv')
    st.subheader('Preprocessor - (OneHotEncoder, StandardScaler)')
    st.download_button('🧹 Download preprocessor',data=open('Models/preprocessor.pkl','rb').read(),file_name='preprocessor.pkl')
    st.subheader('Model - (Gradient Boosting)')
    st.download_button('📦 Download Model',data=open('Models/gradientboosting_model.pkl','rb').read(),file_name='gradientboosting_model.pkl')


if page == "About Project":
    st.header("📄 Project Summary")
    st.markdown("""
    **Business Objective:** Predict whether a customer will purchase a holiday package.

    **Data Source:** Travel.csv

    **Steps Performed:**
    - Data Cleaning and Preprocessing
    - Exploratory Data Analysis
    - Feature Engineering
    - Model Training (Compared 8+ models)
    - Hyperparameter Tuning (RandomizedSearchCV)
    - Final Model: Gradient Boosting Classifier
    - Deployment: Streamlit App

    **Metrics:**
    - Accuracy: 96.5%
    - F1 Score: 0.90
    - ROC-AUC: 0.91

    """)