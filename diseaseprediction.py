import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

st.title("Disease Detection")
tab1,tab2,tab3=st.tabs(['Overview','Lung Cancer Detection','Tumor Size Detection'])
with tab1:
    st.subheader("🩺 Welcome to the Disease Detection System")
    st.write("""
    This web-based application is designed to assist in the early detection and analysis of health conditions using machine learning models.
    
    🧠 **Lung Cancer Detection**: Predicts the likelihood of lung cancer based on lifestyle and symptom inputs such as smoking habits, fatigue, chest pain, etc.
    
    🎯 **Breast Cancer Tumor Size Estimation**: Estimates tumor size in breast cancer patients using clinical features like tumor stage, node involvement, hormone receptor status, and more.
    
    The aim is to provide accessible, AI-powered decision support that can assist in preliminary screening and awareness. Please note that these predictions are not a substitute for professional medical diagnosis.
    """)
with tab2:
    st.header('Lung Cancer Detection')
    df1=pd.read_csv("lung.csv")
    df1.drop("GENDER",axis='columns',inplace=True)
    df1['LUNG_CANCER']=df1['LUNG_CANCER'].apply(lambda x: 1 if x=="YES" else 0)
    x=df1.drop('LUNG_CANCER',axis=True)
    y=df1['LUNG_CANCER']
    model1=LogisticRegression()
    model1.fit(x,y)
    age=st.number_input("Enter Age")
    smoking=st.number_input("Smoking (YES=2,No=1)")
    yellowfing=st.number_input("Yellow Fingers (YES=2,No=1)")
    anxiety=st.number_input("Anxiety (YES=2,No=1)")
    peer=st.number_input("Peer Pressure (YES=2,No=1)")
    chronic=st.number_input("Chronic Disease (YES=2,No=1)")
    fatigue=st.number_input("Fatigue (YES=2,No=1)")
    allergy=st.number_input("Allergy (YES=2,No=1)")
    wheezing=st.number_input("Wheezing (YES=2,No=1)")
    alcohol=st.number_input("Alcohol Consumption (YES=2,No=1)")
    coughing=st.number_input("Coughing (YES=2,No=1)")
    breath=st.number_input("Shortness of Breath (YES=2,No=1)")
    swallow=st.number_input("Difficulty in Swallowing (YES=2,No=1)")
    chest=st.number_input("Chest Pain (YES=2,No=1)")
    pred=model1.predict([[age,smoking,yellowfing,anxiety,peer,chronic,fatigue,allergy,wheezing,alcohol,coughing,breath,swallow,chest]])
    if st.button("Analyse"):
        if pred==1:
            st.warning("Get yourself Checked")
        else:
            st.success("You're healthy")
with tab3:
    st.header("Breast Cancer Tumor Size Prediction")

    df2 = pd.read_csv("encoded_dataset.csv")
    x = df2.drop("Tumor Size", axis="columns")
    y = df2["Tumor Size"]
    model2 = LinearRegression()
    model2.fit(x, y)

    age = st.number_input("Enter Age", step=1)
    tstage = st.number_input("Enter T Stage", step=1)
    nstage = st.number_input("Enter N Stage", step=1)
    sixstage = st.number_input("Enter 6th Stage", step=1)
    differentiate = st.number_input("Enter Differentiate", step=1)
    grade = st.number_input("Enter Grade", step=1)
    astage = st.number_input("Enter A Stage", step=1)
    estrogen = st.number_input("Enter Estrogen Status", step=1)
    progesterone = st.number_input("Enter Progesterone Status", step=1)
    reginol = st.number_input("Enter Regional Node Examined", step=1)
    reginolpos = st.number_input("Enter Regional Node Positive", step=1)

    if st.button("Analyse Tumor Size"):
        try:
            pred2 = model2.predict([[age, tstage, nstage, sixstage, differentiate,
            grade, astage, estrogen, progesterone, reginol, reginolpos]])
            st.success(f"🎯 Predicted Tumor Size: {pred2[0]:.2f}")
        except Exception as e:
            st.error(f"Error: {e}")
