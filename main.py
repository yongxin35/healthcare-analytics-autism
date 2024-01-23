import numpy as np
import streamlit as st
import plotly.express as px
import pickle
import pandas as pd
import base64

# Initialise the score checkboxes state, predict button state, age slider value, and gender radio button
checkbox_state1 = None
checkbox_state2 = None
checkbox_state3 = None
checkbox_state4 = None
checkbox_state5 = None
checkbox_state6 = None
checkbox_state7 = None
checkbox_state8 = None
checkbox_state9 = None
checkbox_state10 = None
button_state = None

age_slider_value = None
gender_radio_button = None

non_asd_probability = None
asd_probability = None


# Create Sidebar
def sidebar_selections():
    global checkbox_state1
    global checkbox_state2
    global checkbox_state3
    global checkbox_state4
    global checkbox_state5
    global checkbox_state6
    global checkbox_state7
    global checkbox_state8
    global checkbox_state9
    global checkbox_state10
    global button_state

    with st.sidebar:
        st.image('BME_Logo.png')

        st.sidebar.success("Select the scores respectively")

        # Add Checkboxes in sidebar (For A1-A10 Score)

        checkbox_state1 = st.checkbox("A1_Score")
        checkbox_state2 = st.checkbox("A2_Score")
        checkbox_state3 = st.checkbox("A3_Score")
        checkbox_state4 = st.checkbox("A4_Score")
        checkbox_state5 = st.checkbox("A5_Score")
        checkbox_state6 = st.checkbox("A6_Score")
        checkbox_state7 = st.checkbox("A7_Score")
        checkbox_state8 = st.checkbox("A8_Score")
        checkbox_state9 = st.checkbox("A9_Score")
        checkbox_state10 = st.checkbox("A10_Score")

        # Add a predict button
        st.write("")
        button_state = st.button("Predict")
        print(button_state)


# Function for predict button
def button_predictions():
    global button_state
    global non_asd_probability
    global asd_probability

    # Import the model
    model = pickle.load(open("model.pkl", "rb"))

    if button_state:

        input_values = input_data()

        non_asd_probability = model.predict_proba(input_values)[0][0]
        asd_probability = model.predict_proba(input_values)[0][1]

        # Round the probabilities to two decimal places
        non_asd_probability_rounded = round(non_asd_probability, 3)
        asd_probability_rounded = round(asd_probability, 3)

        st.write("Probability of not having ASD: ", non_asd_probability_rounded)
        st.write("Probability of having ASD: ", asd_probability_rounded)

    else:
        st.write("(Press the predict button to get a prediction)")


# Create Pie Chart to display the probability
def pie_chart_predictions():
    global button_state

    # Import the model
    model = pickle.load(open("model.pkl", "rb"))

    if button_state:
        input_values1 = input_data()

        non_asd_probability1 = model.predict_proba(input_values1)[0][0]
        asd_probability1 = model.predict_proba(input_values1)[0][1]

        # Round the probabilities to two decimal places
        non_asd_probability_rounded1 = round(non_asd_probability1, 3)
        asd_probability_rounded1 = round(asd_probability1, 3)

        labels = ['ASD', 'Non-ASD']
        prediction_score = [asd_probability_rounded1, non_asd_probability_rounded1]

        fig = px.pie(names=labels, values=prediction_score, title='Pie Chart (ASD Predictions)')

        st.plotly_chart(fig, use_container_width=True)


# Container to show the prediction result
def container_prediction():
    global button_state

    # Import the model
    model = pickle.load(open("model.pkl", "rb"))

    if button_state:

        input_values = input_data()

        prediction = model.predict(input_values)

        if prediction[0] == 0:
            st.success('Negative (Low Probability of having ASD)', icon="✅")

        else:
            st.error('Positive (High Probability of having ASD)', icon="🚨")


# Container description for information
def container_description():
    container1 = st.container(border=True)
    container1.write(
        "This is a fully functional Computer Aided Diagnosis (CAD) that can assist clinicians in making a well-informed diagnosis, but should not be used as a substitute for a professional diagnosis.")

    container2 = st.container(border=True)
    container2.write("The backend Machine Learning Model applied is Logistic Regression."
                     " The inputs that will be used are A1-A10 Scores as well as Gender and Age."
                     " This is based on the dataset Autism Spectrum Quotient (AQ-10 Child).")


# Function to create and download data as CSV
def create_and_download_csv(data):
    # Create a DataFrame
    df = pd.DataFrame(data)

    # Create a download link
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV</a>'

    # Display the download button
    st.markdown(href, unsafe_allow_html=True)


# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = {
        'Name/ID': [],
        'Gender': [],
        'Age': [],
        'A1_Score': [], 'A2_Score': [], 'A3_Score': [], 'A4_Score': [], 'A5_Score': [],
        'A6_Score': [], 'A7_Score': [], 'A8_Score': [], 'A9_Score': [], 'A10_Score': [],
        'ASD_Probability': [],
        'Clinician Notes': []
    }


# Function to read the user inputs, and store them as a list. For feeding into the machine learning model
def input_data():
    global checkbox_state1
    global checkbox_state2
    global checkbox_state3
    global checkbox_state4
    global checkbox_state5
    global checkbox_state6
    global checkbox_state7
    global checkbox_state8
    global checkbox_state9
    global checkbox_state10
    global button_state

    global age_slider_value
    global gender_radio_button

    global asd_probability

    input_data1 = [int(checkbox_state) for checkbox_state in [
        checkbox_state1, checkbox_state2, checkbox_state3, checkbox_state4,
        checkbox_state5, checkbox_state6, checkbox_state7, checkbox_state8,
        checkbox_state9, checkbox_state10
    ]] + [age_slider_value, 1 if gender_radio_button == 'Male' else 0]

    # Convert the list to a numpy array and reshape it
    input_data_array = np.array(input_data1).reshape(1, -1)

    print(input_data_array)

    return input_data_array


# Main app page and program flow
def main():
    global asd_probability
    global non_asd_probability

    global age_slider_value
    global gender_radio_button

    # Add a title
    st.set_page_config(page_title="Autistic Spectrum Disorder (ASD) Screening Tool",
                       page_icon=":health_worker:",
                       layout="wide", )

    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    # Add a header
    st.markdown(
        "<div style='text-align: center;'><h3>Autistic Spectrum Disorder (ASD) Screening Tool</h3></div>",
        unsafe_allow_html=True
    )

    # Add a container description about the ML model
    container_description()

    # Add a sidebar
    sidebar_selections()

    patient_name = st.text_input("Enter Patient's Name/ID:", "")
    st.write(f"Patient's Name/ID: {patient_name}")
    print(patient_name)

    gender_radio_button = st.radio(
        "Gender:",
        ("Male", "Female")
    )

    print(gender_radio_button)

    age_slider_value = st.slider("Age:", min_value=4, max_value=11, value=5, step=1)
    print(age_slider_value)

    col1, col2 = st.columns([4, 1])
    with col1:

        pie_chart_predictions()

    with col2:
        st.write("")
        st.write("")

        st.write("**ASD Predictions:**")
        button_predictions()

        st.write("")
        st.write("")
        st.image('doctor_image.jpg')

    container_prediction()

    clinicians_notes = st.text_input("Enter Clinician's Notes:", "")
    st.write(f"Clinician's Notes: {clinicians_notes}")
    print(clinicians_notes)

    A1_Score = int(checkbox_state1)
    A2_Score = int(checkbox_state2)
    A3_Score = int(checkbox_state3)
    A4_Score = int(checkbox_state4)
    A5_Score = int(checkbox_state5)
    A6_Score = int(checkbox_state6)
    A7_Score = int(checkbox_state7)
    A8_Score = int(checkbox_state8)
    A9_Score = int(checkbox_state9)
    A10_Score = int(checkbox_state10)

    input_values1 = input_data()
    model = pickle.load(open("model.pkl", "rb"))
    asd_probability1 = model.predict_proba(input_values1)[0][1]

    asd = round(asd_probability1, 3)

    print(f"Probability is, {asd}")

    # Store the inputs
    if st.button('Store Data'):
        st.session_state.data['Name/ID'].append(patient_name)
        st.session_state.data['Gender'].append(gender_radio_button)
        st.session_state.data['Age'].append(age_slider_value)
        st.session_state.data['A1_Score'].append(A1_Score)
        st.session_state.data['A2_Score'].append(A2_Score)
        st.session_state.data['A3_Score'].append(A3_Score)
        st.session_state.data['A4_Score'].append(A4_Score)
        st.session_state.data['A5_Score'].append(A5_Score)
        st.session_state.data['A6_Score'].append(A6_Score)
        st.session_state.data['A7_Score'].append(A7_Score)
        st.session_state.data['A8_Score'].append(A8_Score)
        st.session_state.data['A9_Score'].append(A9_Score)
        st.session_state.data['A10_Score'].append(A10_Score)
        st.session_state.data['ASD_Probability'].append(asd)
        st.session_state.data['Clinician Notes'].append(clinicians_notes)

        print(st.session_state.data)

    # Create and download CSV button
    if st.button('Download CSV'):
        create_and_download_csv(st.session_state.data)


if __name__ == '__main__':
    main()

# To run this, on the terminal type: streamlit run main.py
