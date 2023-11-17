import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats



st.write("""
         #This app predicts the **Customer Churning**!
         """)




#Fetching the model
#model = pickle.load(open('trained_model.sav','rb'))
# Loading the model
loaded_model = joblib.load('CustomerChurning.joblib')
scaler = joblib.load('scaler.joblib')


def predict(TotalCharges,MonthlyCharges,tenure,Contract,PaymentMethod,
                                OnlineSecurity,TechSupport,OnlineBackup,PaperlessBilling,Partner,InternetSecurity):
    feature1=float(TotalCharges)
    feature2=float(MonthlyCharges),
    feature3=float(tenure)
    feature4=float(tenure)
    feature5=float(PaymentMethod)
    feature6=float(OnlineSecurity)
    feature7=float( TechSupport)
    feature8=float(OnlineBackup)
    feature9=float(PaperlessBilling)
    feature10=float(Partner)
    feature11=float(InternetSecurity)
    
    #Pre-processing the data
    #Creating a user_input list
    #inputs = [feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9,feature10]

    # Assuming inputs is a list or array of input features
    inputs = np.array([TotalCharges, MonthlyCharges, tenure, Contract,PaymentMethod, OnlineSecurity,TechSupport,OnlineBackup,
                       PaperlessBilling,Partner,InternetSecurity])

    # Reshape the array if needed
    inputs = inputs.reshape(1, -1)  # Adjust the shape based on your data

    
    
    #Scaling the features 

    scaled_features= scaler.transform(inputs)

    #Prediction

    prediction = loaded_model.predict(scaled_features)[0] 

    confidence_factor=prediction.squeeze()

    #Threshold
    if prediction >= 0.5:
        result="Yes"
    else:
         result="No"

    #Calculating the confidence score
    std_error = 0.1 
    error_margin= std_error * stats.t.ppf((1+0.95)/2,7043-1)  
    confidence_level = 95


    interval= (round(prediction-error_margin,2),round(prediction+error_margin,2))  
    
    return result, confidence_factor,interval







def main():
    st.title("Customer Churning Prediction")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Customer Churning Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    

    #Taking in the features for the prediction.
    
    TotalCharges = st.text_input("TotalCharges")
    MonthlyCharges= st.text_input("MonthlyCharges")
    tenure = st.text_input("tenure")
    Contract = st.text_input("Contract (0-Month-to-month,1-One year,2-Two year)")
    OnlineSecurity = st.text_input("Onlinesecurity (0-No,1-Yes,2-No internet service)")
    InternetSecurity = st.text_input("InternetService (0-DSL,1-Fiber optic,2-No)")
    TechSupport= st.text_input("TechSupport (0-No,1-Yes,2-No internet service)")
    PaperlessBilling = st.text_input("PaperlessBilling (1-Yes,0- No)")
    PaymentMethod = st.text_input('PaymentMethod (0-Electronic check,1-Mailed check,2-Bank transfer(automatic),3-Credit card (automatic))')
    Partner = st.text_input("Partner (1-Yes,0-No)")
    OnlineBackup = st.text_input("OnlineBackup (1-Yes,0-No,2-No internet Service)")
    
    

    if st.button("Predict"):
        output, confidence_score,interval= predict(TotalCharges, MonthlyCharges, tenure, Contract,PaymentMethod, OnlineSecurity,TechSupport,OnlineBackup,
                       PaperlessBilling,Partner,InternetSecurity)
        
        st.success(f"Customer Churn: {output}")
        st.info(f"Confidence Factor: {confidence_score}")
        st.info(f"Confidence interval: {interval}")


if __name__=='__main__':
        main()


    


