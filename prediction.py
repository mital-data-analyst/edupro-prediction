import pandas as pd

# Excel file name
file_name = "EduPro Online Platform.xlsx"

# Load sheets
courses = pd.read_excel(file_name, sheet_name="Courses")

teachers = pd.read_excel(file_name, sheet_name="Teachers")

transactions = pd.read_excel(file_name, sheet_name="Transactions")

users = pd.read_excel(file_name, sheet_name="Users")

# Print Data
print("COURSES DATA")
print(courses.head())

print("TEACHERS DATA")
print(teachers.head())

print("TRANSACTIONS DATA")
print(transactions.head())

print("USERS DATA")
print(users.head())


# =========================
# MERGE TABLES
# =========================

# Merge Transactions + Courses
data = transactions.merge(courses, on="CourseID")

# Merge Teachers
data = data.merge(teachers, on="TeacherID")

print("FINAL MERGED DATA")

print(data.head())

# =========================
# REVENUE CALCULATION
# =========================

revenue = data.groupby("CourseID")["Amount"].sum()

print("COURSE REVENUE")

print(revenue)

# =========================
# ENROLLMENT COUNT
# =========================

enrollment = data.groupby("CourseID").size()

print("ENROLLMENT COUNT")

print(enrollment)

# =========================
# FINAL DATASET
# =========================

# Revenue DataFrame
revenue_df = data.groupby("CourseID")["Amount"].sum().reset_index()

# Enrollment DataFrame
enrollment_df = data.groupby("CourseID").size().reset_index(name="EnrollmentCount")

# Merge with Courses
final_data = courses.merge(revenue_df, on="CourseID")

final_data = final_data.merge(enrollment_df, on="CourseID")

print("FINAL DATASET")

print(final_data.head())


# =========================
# ENCODING
# =========================

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Convert text to numbers
final_data["CourseCategory"] = le.fit_transform(final_data["CourseCategory"])

final_data["CourseLevel"] = le.fit_transform(final_data["CourseLevel"])

print("ENCODED DATA")

print(final_data.head())

print(final_data[["CourseCategory", "CourseLevel"]].head())

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

y = final_data["Amount"]

print("FEATURES")

print(X.head())

print("TARGET")

print(y.head())

# =========================
# TRAIN TEST SPLIT
# =========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("TRAINING DATA SIZE")

print(X_train.shape)

print("TESTING DATA SIZE")

print(X_test.shape)

# =========================
# MACHINE LEARNING MODEL
# =========================

from sklearn.ensemble import RandomForestRegressor

# Create Model
model = RandomForestRegressor()

# Train Model
model.fit(X_train, y_train)

print("MODEL TRAINED SUCCESSFULLY")

# =========================
# PREDICTION
# =========================

prediction = model.predict([
    [3000, 10, 4.5, 9, 1]
])

print("PREDICTED REVENUE")

print(prediction)

# =========================
# MODEL ACCURACY
# =========================

from sklearn.metrics import r2_score

# Predict Test Data
y_pred = model.predict(X_test)

# Accuracy Score
score = r2_score(y_test, y_pred)

print("MODEL ACCURACY")

print(score)