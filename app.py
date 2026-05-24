import streamlit as st
import pandas as pd


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# add kari

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# =========================
# PAGE TITLE
# =========================

st.title("EduPro AI Prediction System")
# =========================
# LOAD DATA
# =========================

file_name = "EduPro Online Platform.xlsx"

courses = pd.read_excel(file_name, sheet_name="Courses")

teachers = pd.read_excel(file_name, sheet_name="Teachers")

transactions = pd.read_excel(file_name, sheet_name="Transactions")

# =========================
# MERGE DATA
# =========================

data = transactions.merge(courses, on="CourseID")

data = data.merge(teachers, on="TeacherID")

# =========================
# FINAL DATASET
# =========================

revenue_df = data.groupby("CourseID")["Amount"].sum().reset_index()

enrollment_df = data.groupby("CourseID").size().reset_index(name="EnrollmentCount")

final_data = courses.merge(revenue_df, on="CourseID")

final_data = final_data.merge(enrollment_df, on="CourseID")

# =========================
# SAVE ORIGINAL CATEGORY NAMES
# =========================

category_names = final_data["CourseCategory"].unique()

level_names = final_data["CourseLevel"].unique()

# =========================
# ENCODING
# =========================

category_encoder = LabelEncoder()

level_encoder = LabelEncoder()

final_data["CourseCategory"] = category_encoder.fit_transform(final_data["CourseCategory"])

final_data["CourseLevel"] = level_encoder.fit_transform(final_data["CourseLevel"])

# =========================
# FEATURES & TARGET
# =========================

X = final_data[
[
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "CourseCategory",
    "CourseLevel"
]
]

# TARGET = REVENUE
y = final_data["Amount"]

# FOR ENROLLMENT PREDICTION
# y = final_data["EnrollmentCount"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = RandomForestRegressor()
import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="EduPro AI System",
    layout="centered"
)

# =========================
# PAGE TITLE
# =========================

st.title("EduPro AI Prediction System")

# =========================
# LOAD DATA
# =========================

file_name = "EduPro Online Platform.xlsx"

courses = pd.read_excel(file_name, sheet_name="Courses")

teachers = pd.read_excel(file_name, sheet_name="Teachers")

transactions = pd.read_excel(file_name, sheet_name="Transactions")

# =========================
# MERGE DATA
# =========================

data = transactions.merge(courses, on="CourseID")

data = data.merge(teachers, on="TeacherID")

# =========================
# FINAL DATASET
# =========================

revenue_df = data.groupby("CourseID")["Amount"].sum().reset_index()

enrollment_df = data.groupby("CourseID").size().reset_index(name="EnrollmentCount")

final_data = courses.merge(revenue_df, on="CourseID")

final_data = final_data.merge(enrollment_df, on="CourseID")

# =========================
# SAVE ORIGINAL CATEGORY NAMES
# =========================

category_names = final_data["CourseCategory"].unique()

level_names = final_data["CourseLevel"].unique()

# =========================
# ENCODING
# =========================

category_encoder = LabelEncoder()

level_encoder = LabelEncoder()

final_data["CourseCategory"] = category_encoder.fit_transform(
    final_data["CourseCategory"]
)

final_data["CourseLevel"] = level_encoder.fit_transform(
    final_data["CourseLevel"]
)

# =========================
# FEATURES & TARGET
# =========================

X = final_data[
    [
        "CoursePrice",
        "CourseDuration",
        "CourseRating",
        "CourseCategory",
        "CourseLevel"
    ]
]

# TARGET = REVENUE
y = final_data["Amount"]

# FOR ENROLLMENT PREDICTION
# y = final_data["EnrollmentCount"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

model = RandomForestRegressor()

model.fit(X_train, y_train)

# =========================
# MODEL NAME
# =========================

st.subheader("Model Information")

st.write("Model Used: Random Forest Regressor")

# =========================
# MODEL EVALUATION
# =========================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

r2 = r2_score(y_test, y_pred)

st.subheader("Model Accuracy")

st.write("MAE:", round(mae, 2))

st.write("RMSE:", round(rmse, 2))

st.write("R² Score:", round(r2, 2))

# =========================
# FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

st.subheader("Feature Importance")

st.bar_chart(
    importance.set_index("Feature")
)

# =========================
# USER INPUTS
# =========================

st.header("Enter Course Details")

price = st.number_input(
    "Course Price",
    min_value=0.0
)

duration = st.number_input(
    "Course Duration",
    min_value=0.0
)

rating = st.number_input(
    "Course Rating",
    min_value=0.0,
    max_value=5.0
)

# CATEGORY DROPDOWN

selected_category = st.selectbox(
    "Course Category",
    category_names
)

# LEVEL DROPDOWN

selected_level = st.selectbox(
    "Course Level",
    level_names
)

# ENCODE USER INPUT

category = category_encoder.transform(
    [selected_category]
)[0]

level = level_encoder.transform(
    [selected_level]
)[0]

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Revenue", key="btn1"):

    prediction = model.predict([
        [price, duration, rating, category, level]
    ])

    st.success(
        f"Predicted Revenue: {prediction[0]:,.2f}"
    )

# =========================
# REVENUE CHART
# =========================

st.header("Course Revenue Chart")

st.bar_chart(final_data["Amount"])

# =========================
# ENROLLMENT CHART
# =========================

st.header("Enrollment Count Chart")

st.bar_chart(final_data["EnrollmentCount"])

# =========================
# SHOW DATA
# =========================

st.header("Final Dataset")

st.dataframe(final_data)

model.fit(X_train, y_train)
# add kari aa  line
st.write("Model Used: Random Forest Regressor")
# =========================
# USER INPUTS
# =========================



# ENCODE USER INPUT
category = category_encoder.transform([selected_category])[0]

level = level_encoder.transform([selected_level])[0]

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict Revenue", key="btn2"):

    prediction = model.predict([
        [price, duration, rating, category, level]
    ])

    st.success(f"Predicted Revenue: {prediction[0]:,.2f}")

# =========================
# REVENUE CHART
# =========================

st.header("Course Revenue Chart")

st.bar_chart(final_data["Amount"])

# =========================
# ENROLLMENT CHART
# =========================

st.header("Enrollment Count Chart")

st.bar_chart(final_data["EnrollmentCount"])

# =========================
# SHOW DATA
# =========================

st.header("Final Dataset")

st.dataframe(final_data)
