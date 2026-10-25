import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


st.title('Titanic Survival Prediction')
st.write('An Artificial Neural Network model that predicts the survival chances on basis of passenger details')
Pclass=st.slider('Enter the Class in which passenger was travelling',1,3)
Sex=st.radio('Gender',['male','female'])
Age=st.number_input('Enter the Age of passenger',0,100)
SibSp=st.slider('Number of Sibling/Spouse',0,8)
Parch=st.slider('Total Parents or Childern travelling',0,6)
Fare=st.number_input('Enter the Fare paid by passenger',0.0,500.0)
Embarked=st.selectbox('Port of Embarkation',['Cherbourg','Queenstown','Southampton'])

data=pd.DataFrame([{
    'Pclass': Pclass,
    'Sex': Sex,
    'Age': Age,
    'SibSp': SibSp,
    'Parch': Parch,
    'Fare': Fare,
    'Embarked': Embarked
}])
if st.button('Passenger Details'):
    st.write(data)

model=load_model('titanic_model.h5')
with open('label.pkl','rb') as f:
    label=pickle.load(f)

with open('ohe.pkl','rb') as f:
    ohe=pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler=pickle.load(f)

data['Sex']=label.transform(data['Sex'])
new=ohe.transform(data[['Embarked']])
new=pd.DataFrame(new,columns=['Embarked_C','Embarked_Q','Embarked_S'])
data=pd.concat([data,new],axis=1)
data.drop(['Embarked'],axis=1,inplace=True)
cols=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
data[cols] = scaler.transform(data[cols])
y=model.predict(data)
def chance(y):
    if y[0][0]>0.5:
        return 'There may be a chance of survival'
    else:
        return'God May bless you'

if st.button('Confirm Details'):
    st.write('The percentage of survival is',y*100)
    st.write(chance(y))