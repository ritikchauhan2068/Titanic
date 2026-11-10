import streamlit as st
import pandas as pd
import pickle
import os


# -----------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Survival Prediction")
st.markdown(
    "### Random Forest Classification"
)

st.write(
    "Predict passenger survival using a pre-trained "
    "Machine Learning model."
)


# -----------------------------------------
# 2. LOAD PICKLE MODEL
# -----------------------------------------

@st.cache_resource
def load_model():

    with open("titanic_model11.pkl", "rb") as file:
        model = pickle.load(file)

    return model


try:
    model = load_model()
    st.sidebar.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# -----------------------------------------
# 3. MODEL INFORMATION
# -----------------------------------------

st.header("1. Model Information")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Algorithm",
    "Random Forest"
)

col2.metric(
    "Model Type",
    "Classification"
)

col3.metric(
    "Model Status",
    "Loaded"
)

st.write("**Input Features:**")

st.code(
    "Age, Fare, SibSp, Parch, Sex, Embarked, Pclass"
)


# -----------------------------------------
# 4. PASSENGER INPUT FORM
# -----------------------------------------

st.header("2. Passenger Information")

st.write(
    "Enter the passenger details to predict survival."
)

with st.form("passenger_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3]
        )

        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

        age = st.number_input(
            "Age",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

    with col2:

        fare = st.number_input(
            "Fare",
            min_value=0.0,
            max_value=600.0,
            value=7.75
        )

        embarked = st.selectbox(
            "Port of Embarkation",
            ["S", "C", "Q"]
        )

        sibsp = st.number_input(
            "Siblings / Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0
        )

    with col3:

        parch = st.number_input(
            "Parents / Children Aboard",
            min_value=0,
            max_value=10,
            value=0
        )

    predict_button = st.form_submit_button(
        "Predict Survival",
        type="primary"
    )


# -----------------------------------------
# 5. CREATE INPUT DATAFRAME
# -----------------------------------------

if predict_button:

    passenger_data = pd.DataFrame({

        "Age": [age],

        "Fare": [fare],

        "SibSp": [sibsp],

        "Parch": [parch],

        "Sex": [sex],

        "Embarked": [embarked],

        "Pclass": [pclass]

    })


    # -------------------------------------
    # 6. MAKE PREDICTION
    # -------------------------------------

    try:

        prediction = model.predict(
            passenger_data
        )[0]

        probability = model.predict_proba(
            passenger_data
        )[0]


        survival_probability = probability[1] * 100

        non_survival_probability = probability[0] * 100


        # ---------------------------------
        # 7. DISPLAY RESULTS
        # ---------------------------------

        st.divider()

        st.header("3. Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.success(
                    "🚢 Passenger is predicted to SURVIVE"
                )

            else:

                st.error(
                    "Passenger is predicted NOT TO SURVIVE"
                )


        with col2:

            st.metric(
                "Survival Probability",
                f"{survival_probability:.2f}%"
            )


        # ---------------------------------
        # 8. PROBABILITY VISUALIZATION
        # ---------------------------------

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({

            "Outcome": [
                "Did Not Survive",
                "Survived"
            ],

            "Probability": [
                non_survival_probability,
                survival_probability
            ]

        })

        st.bar_chart(
            probability_df.set_index("Outcome")
        )


        # ---------------------------------
        # 9. PASSENGER DETAILS
        # ---------------------------------

        st.subheader("Passenger Details")

        st.dataframe(
            passenger_data,
            use_container_width=True
        )


        st.info(
            "Prediction generated using your pre-trained "
            "Random Forest classification pipeline."
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# -----------------------------------------
# 10. FOOTER
# -----------------------------------------

st.divider()

st.markdown(
    """
    **Titanic Survival Prediction System**

    Developed using:

    - Python
    - Pandas
    - Scikit-learn
    - Random Forest
    - Streamlit
    """
)