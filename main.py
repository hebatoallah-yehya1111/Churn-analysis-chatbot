import streamlit as st
from langchain_ollama import OllamaLLM
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
import xgboost as xgb
import joblib



with open('xgboost_model.pkl', 'rb') as f:
    ML_model = pickle.load(f)
    print(type(ML_model))

with open('scalar.pkl', 'rb') as f:
    scaler = joblib.load(f)
    print(type(scaler))

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = joblib.load(f)
    print(type(label_encoder))

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = joblib.load(f)
    print(type(one_hot_encoder))


# Instantiate the model
model = OllamaLLM(model="llama3")

# Define the valid options for each feature
valid_answers = {
    'gender': ['Female', 'Male'],
    'Senior_Citizen': ['Yes', 'No'],
    'Is_Married': ['Yes', 'No'],
    'Dependents': ['No', 'Yes'],
    'Phone_Service': ['No', 'Yes'],
    'Dual': ['No phone service', 'No', 'Yes'],
    'Internet_Service': ['DSL', 'Fiber optic', 'No'],
    'Online_Security': ['No', 'Yes', 'No internet service'],
    'Online_Backup': ['Yes', 'No', 'No internet service'],
    'Device_Protection': ['No', 'Yes', 'No internet service'],
    'Tech_Support': ['No', 'Yes', 'No internet service'],
    'Streaming_TV': ['No', 'Yes', 'No internet service'],
    'Streaming_Movies': ['No', 'Yes', 'No internet service'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'Paperless_Billing': ['Yes', 'No'],
    'Payment_Method': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    'tenure': ['Numeric value representing months'],
    'Monthly_Charges': ['Numeric value representing monthly charge'],
    'Total_Charges': ['Numeric value representing total charges'],
    'Churn': ['Yes', 'No']
}

# Define the feature questions
feature_questions = [
    {"key": "gender", "question": "Can you tell us if the customer is male or female?"},
    {"key": "Senior_Citizen", "question": "Is the customer a senior citizen?"},
    {"key": "Is_Married", "question": "Is the customer married?"},
    {"key": "Dependents", "question": "Does the customer have any dependents?"},
    {"key": "tenure", "question": "How long has the customer been with us (in months)?"},
    {"key": "Phone_Service", "question": "Does the customer use our phone service?"},
    {"key": "Dual", "question": "Does the customer have both phone and internet services?"},
    {"key": "Internet_Service", "question": "What type of internet service does the customer use?"},
    {"key": "Online_Security", "question": "Does the customer have online security services?"},
    {"key": "Online_Backup", "question": "Is the customer using online backup services?"},
    {"key": "Device_Protection", "question": "Does the customer have device protection?"},
    {"key": "Tech_Support", "question": "Does the customer have access to tech support?"},
    {"key": "Streaming_TV", "question": "Does the customer enjoy streaming TV?"},
    {"key": "Streaming_Movies", "question": "Does the customer watch streaming movies?"},
    {"key": "Contract", "question": "What type of contract does the customer have?"},
    {"key": "Paperless_Billing", "question": "Is the customer using paperless billing?"},
    {"key": "Payment_Method", "question": "How does the customer prefer to pay?"},
    {"key": "Monthly_Charges", "question": "What are the customer's monthly charges?"},
    {"key": "Total_Charges", "question": "What are the customer's total charges so far?"}
]

def standardize_with_ollama(feature, question, response):
    """
    Standardize the response using Ollama model
    """
    # Get the valid options for the feature
    valid_options = valid_answers.get(feature, [])

    # Create the prompt to ask Llama to standardize the response
    prompt = f"""
    You are a helpful assistant. Based on the following question and response, please return only the valid standardized answer from the following list:

    Feature: {feature}
    Valid options: {', '.join(valid_options)}

    Question: {question}
    Response: {response}

    Please return the standardized answer, nothing else. No extra explanation.

    """
    
    # Generate the response using Llama
    try:
        result = model.invoke(input=prompt)
        
        # Clean the result to remove any extra explanation or spaces
        standardized_answer = result.strip().split('\n')[0]
        
        return standardized_answer
    except Exception as e:
        st.error(f"Error in standardization: {e}")
        return response

def chatbot_questionnaire():
    # Initialize session state
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'responses' not in st.session_state:
        st.session_state.responses = {}

    # Check if all questions are answered
    if st.session_state.current_question >= len(feature_questions):
        st.success("Questionnaire Complete!")
        
        # Create a DataFrame from the responses
        df = pd.DataFrame([st.session_state.responses])
        
        # Display summary of responses
        st.write("### Responses Summary:")
        st.dataframe(df)
        # Convert to numeric
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
        df['Monthly_Charges'] = pd.to_numeric(df['Monthly_Charges'], errors='coerce')
        df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
        # Step 1: Filter categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        print("Categorical columns:", categorical_columns)

        # Step 1: Identify binary columns and categorical columns
        binary_columns = [col for col in categorical_columns if set(df[col].unique()) <= {'Yes', 'No'}]
        non_binary_columns  = [col for col in categorical_columns if col not in binary_columns]

        # Step 2: Apply LabelEncoder for binary columns
        for col in binary_columns:
          df[col] = label_encoder.transform(df[col])

        encoded = one_hot_encoder.transform(df[non_binary_columns]) # Fit using non_binary_columns

       # Step 4: Add one-hot encoded columns back to the DataFrame
        encoded_df = pd.DataFrame(encoded) # Get feature names using non_binary_columns
      #Join encoded_df with original df
        df = df.drop(columns=non_binary_columns)  # Drop non_binary_columns
        df = df.reset_index(drop=True) # Reset index for both DataFrames
        encoded_df = encoded_df.reset_index(drop=True) # Reset index for both DataFrames
        df = pd.concat([df, encoded_df], axis=1)
        df[['tenure', 'Monthly_Charges', 'Total_Charges']] = scaler.transform(df[['tenure', 'Monthly_Charges', 'Total_Charges']])
        predictions =ML_model.predict(df)
        st.write(f"The Predictions of the Machine Learning Churn Model : {predictions}")

        
        # Option to start over
        if st.button("Start Over"):
            # Reset session state
            st.session_state.current_question = 0
            st.session_state.responses = {}
            st.rerun()
        return

    # Get the current question
    current_q = feature_questions[st.session_state.current_question]

    # Display progress
    progress = (st.session_state.current_question / len(feature_questions)) * 100
    st.progress(progress / 100)

    # Show valid options
    if current_q['key'] in valid_answers:
        st.info(f"Valid options for {current_q['key']}: {', '.join(valid_answers[current_q['key']])}")

    # Assistant's message
    with st.chat_message("assistant"):
        st.write(current_q["question"])

    # User input
    user_response = st.chat_input("Type your answer here")

    # Process the response
    if user_response:
        # Standardize the response using Ollama
        standardized_response = standardize_with_ollama(
            current_q['key'], 
            current_q["question"], 
            user_response
        )

        # Display the user's response
        with st.chat_message("user"):
            st.write(user_response)

        # Display the standardized response
        with st.chat_message("assistant"):
            st.write(f"Standardized answer: {standardized_response}")

        # Store the standardized response
        st.session_state.responses[current_q['key']] = standardized_response

        # Move to the next question
        st.session_state.current_question += 1

        # Rerun to show the next question
        st.rerun()


def main():
    st.title("Welcome to churn analysis assistant")
    chatbot_questionnaire()

if __name__ == "__main__":
    main()
    