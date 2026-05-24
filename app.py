import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
# =========================
# PAGE SETTINGS
# =========================
st.set_page_config(
    page_title="EduPro AI System",
    layout="centered"
)

# =========================
# PAGE TITLE (Ab Sirf Ek Baar)
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
# FINAL DATASET CREATION
# =========================
revenue_df = data.groupby("CourseID")["Amount"].sum().reset_index()
enrollment_df = data.groupby("CourseID").size().reset_index(name="EnrollmentCount")

final_data = courses.merge(revenue_df, on="CourseID")
final_data = final_data.merge(enrollment_df, on="CourseID")

# Save original names for dropdowns
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
# FEATURES & TARGET (Revenue)
# =========================
X = final_data[["CoursePrice", "CourseDuration", "CourseRating", "CourseCategory", "CourseLevel"]]
y = final_data["Amount"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# MODEL TRAINING
# =========================
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# =========================
# MODEL INFORMATION DISPLAY
# =========================
st.subheader("Model Information")
st.write("Model Used: Random Forest Regressor")

# Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

st.subheader("Model Accuracy")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{round(mae, 2)}")
col2.metric("RMSE", f"{round(rmse, 2)}")
col3.metric("R² Score", f"{round(r2, 2)}")

# Feature Importance
st.subheader("Feature Importance")
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).set_index("Feature")
st.bar_chart(importance)

# =========================
# USER INPUTS FOR PREDICTION
# =========================
st.header("Enter Course Details")

price = st.number_input("Course Price", min_value=0.0, value=3000.0)
duration = st.number_input("Course Duration (in Hours)", min_value=0.0, value=12.0)
rating = st.number_input("Course Rating", min_value=0.0, max_value=5.0, value=4.5)
selected_category = st.selectbox("Course Category", category_names)
selected_level = st.selectbox("Course Level", level_names)

# Encode user input for prediction
category = category_encoder.transform([selected_category])[0]
level = level_encoder.transform([selected_level])[0]

# Prediction Button (Sirf Ek Baar)
if st.button("Predict Revenue", key="predict_btn"):
    prediction = model.predict([[price, duration, rating, category, level]])
    st.success(f"💰 Predicted Revenue: {prediction[0]:,.2f}")

# =========================
# VISUALIZATIONS & DATA
# =========================
st.header("Course Revenue Chart")
st.bar_chart(final_data.set_index("CourseName")["Amount"])

st.header("Enrollment Count Chart")
st.bar_chart(final_data.set_index("CourseName")["EnrollmentCount"])

st.header("Final Dataset")
st.dataframe(final_data)
