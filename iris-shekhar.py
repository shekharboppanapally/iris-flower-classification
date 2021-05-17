#installing all the libraries required
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

def user_input(): #here we are taking inputs from the users using sidebars with the help of sliders
  sepal_length=st.sidebar.slider('sepal length',4.3,7.9,5.4) #(min=4.3,max=7.9,initial=5.4)
  sepal_width=st.sidebar.slider('sepal width',2.0,4.4,4.0)
  petal_length=st.sidebar.slider('petal length',1.0,6.9,4.4)
  petal_width=st.sidebar.slider('petal width',0.1,2.5,0.5)
  data={'sepal length':sepal_length,
        'sepal width':sepal_width,
        'petal length':petal_length,
        'petal width':petal_width}

  features=pd.DataFrame(data,index=[0])
  return features

st.title(' **IRIS FLOWER CLASSIFICATION** ')
st.write('#  using IRIS dataset by_SHEKHAR')
st.write('number of classes:3') #3 classes are:= 1.setosa,2.verginica,3.versicolor
st.write('classifier:KNN')
st.write('Accuracy :0.95') 
st.sidebar.header('User input parameters')
st.subheader('User input parameters')

df=user_input()
st.write(df)

iris=datasets.load_iris()
x=iris.data
y=iris.target

#KNN algorithm
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x,y)
prediction=model.predict(df)

st.subheader('class labels and corresponding index number')
st.write(iris.target_names)

#predicting the output
st.subheader('prediction')
st.write(iris.target_names[prediction])