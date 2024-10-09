import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


st.markdown(
    """
    <style>
    .title {
        font-size: 20px;
        font-weight: 600;
    }
    .intro  {
        font-size: 30px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True
)


st.markdown('<p class ="intro">Pima Indians Diabetes Data Exploration</p>', unsafe_allow_html=True)
st.write("""
This report explores the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) from Kaggle. 
The dataset contains information about female patients of at least 21 years old from the Pima Indian population, 
where the goal is to predict diabetes outcomes based on health metrics.
""")

st.markdown('<hr>', unsafe_allow_html=True)

df = pd.read_csv('diabetes.csv')

df.fillna(df.median(), inplace=True)

outliers_cols = ['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'BloodPressure', 'SkinThickness']
for col in outliers_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    Lower_Fence = Q1 - 1.5 * IQR
    Upper_Fence = Q3 + 1.5 * IQR
    df[col] = df[col].apply(lambda x: Lower_Fence if x < Lower_Fence else (Upper_Fence if x > Upper_Fence else x))

st.sidebar.header("Graph and Chart Visualization")
chart_type = st.sidebar.selectbox(
    'Select a chart type:',
    ['Histogram', 'Box Plot', 'Correlation Heatmap']
)

if chart_type == 'Histogram':
    feature = st.sidebar.selectbox('Select a column for the Histogram:', df.columns)
    st.markdown(f'<p class="title">Histogram for {feature}</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)

    st.markdown('<p class="intro">Analysis</p>', unsafe_allow_html=True)
    if feature == 'Pregnancies':
        st.write("""
        The histogram for Pregnancies indicates the distribution of the number of pregnancies among patients. 
        A significant number of patients have between 0 to 3 pregnancies, suggesting a common pattern. 
        Understanding this can help in predicting diabetes risk related to pregnancy history.
        """)
    elif feature == 'Glucose':
        st.write("""
        The histogram for Glucose levels shows a right-skewed distribution, with many patients having 
        glucose levels below the average. Higher glucose levels are a known risk factor for diabetes, 
        and this visualization highlights the prevalence of elevated glucose in some individuals.
        """)
    elif feature == 'Insulin':
        st.write("""
        The histogram for Insulin levels reveals a concentration of patients with lower insulin levels, 
        indicating potential insulin resistance. Monitoring insulin levels can be crucial in diabetes prediction.
        """)
    elif feature == 'BMI':
        st.write("""
        The histogram for BMI demonstrates that most patients fall within the overweight category, 
        which may contribute to higher diabetes risk. This emphasizes the importance of weight management 
        in diabetes prevention strategies.
        """)
    elif feature == 'DiabetesPedigreeFunction':
        st.write("""
        The histogram for Diabetes Pedigree Function indicates that many patients have low scores, 
        suggesting fewer genetic predispositions to diabetes. However, higher scores still warrant attention 
        for diabetes risk evaluation.
        """)
    elif feature == 'Age':
        st.write("""
        The histogram for Age illustrates that the majority of patients are in the middle-aged group, 
        which aligns with higher diabetes incidence rates in this demographic. Age is a significant factor 
        in diabetes risk assessment.
        """)
    elif feature == 'BloodPressure':
        st.write("""
        The histogram for Blood Pressure shows a peak at lower values, indicating many patients have 
        normal blood pressure levels. Monitoring blood pressure is essential, as it can influence diabetes 
        outcomes.
        """)
    elif feature == 'SkinThickness':
        st.write("""
        The histogram for Skin Thickness illustrates the distribution of skin fold thickness measurements. 
        A concentration of low measurements may suggest a lower risk of diabetes, but values above average 
        should be closely monitored.
        """)

elif chart_type == 'Box Plot':
    feature = st.sidebar.selectbox('Select a column for the Box Plot:', df.columns)
    st.markdown(f'<p class="title">Box Plot for {feature}</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    sns.boxplot(y=df[feature], ax=ax)
    st.pyplot(fig)

    st.markdown('<p class="intro">Analysis</p>', unsafe_allow_html=True)
    st.write(f"""
    The box plot for {feature} shows the spread of data, highlighting the median, variability, and outliers. 
    Extreme values may indicate patients with unusual conditions that could influence diabetes outcomes.
    """)

elif chart_type == 'Correlation Heatmap':
    st.markdown('<p class="title">Correlation Heatmap </p>', unsafe_allow_html=True)
    correlation_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown('<p class="intro">Analysis</p>', unsafe_allow_html=True)
    st.write("""
    The heatmap shows how different features relate to each other. Strong correlations, such as between glucose and BMI, 
    indicate potential links to diabetes risk. These insights help in selecting important features for diabetes prediction models.
    """)

st.markdown('<hr>', unsafe_allow_html=True)

st.markdown('<p class="intro">Conclusion</p>', unsafe_allow_html=True)

st.write("""
In this exploration of the Pima Indians Diabetes Dataset, several key insights were identified:

1. **Pregnancy and Diabetes Risk**: The data shows that the number of pregnancies is an important factor in assessing diabetes risk. Many patients with multiple pregnancies exhibited higher chances of diabetes, suggesting that pregnancy-related changes in the body can influence long-term health outcomes.

2. **Glucose Levels as a Major Indicator**: The dataset reveals that glucose levels are a critical predictor of diabetes. Elevated glucose levels were observed in many patients with diabetes, reaffirming its role as a primary risk factor.

3. **BMI and Weight Management**: A significant portion of the patients had high BMI, correlating with a greater risk of developing diabetes. This finding highlights the importance of maintaining a healthy weight to reduce diabetes risks.

4. **Age as a Contributing Factor**: Age was strongly associated with diabetes incidence, with older patients showing a higher prevalence of diabetes. This suggests that age, along with other factors like genetics and lifestyle, plays a significant role in diabetes onset.

5. **Insulin Levels and Resistance**: Many patients with lower insulin levels or potential insulin resistance were identified, reinforcing the connection between insulin regulation and diabetes development.

6. **Correlation Insights**: The correlation heatmap highlighted key relationships between features such as glucose, BMI, and age. These relationships suggest that glucose and BMI are intertwined in predicting diabetes risk, and age amplifies the overall risk profile.

### Key Takeaways:
- Health metrics such as glucose, BMI, insulin levels, and age are strong indicators of diabetes risk.
- Lifestyle and genetic predisposition, as seen in factors like BMI and the Diabetes Pedigree Function, significantly influence diabetes outcomes.
- Early detection and management of risk factors, including weight management and glucose regulation, are crucial for diabetes prevention.

Overall, this exploration provides a deeper understanding of the key factors driving diabetes risk in this population. These insights can guide future predictive modeling and help focus efforts on prevention strategies.
""")
