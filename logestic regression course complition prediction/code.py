import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

student_data=pd.read_csv('student_course_data.csv')

X=student_data.drop(columns='target')
Y=student_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2,random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

# checking accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('\nAccuracy on Training data :', training_data_accuracy)

# check accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test data     :', test_data_accuracy)

#example
#(study_hours, attendance, assignments, past_perf, motivation)
input_data = [[25, 90, 8, 80, 9]]  # 2D list 

# predict
prediction = model.predict(input_data)
print("\nPrediction value:", prediction)

if prediction[0] == 0:
    print('The student is likely to NOT complete the course.')
else:
    print('The student is likely to COMPLETE the course.')