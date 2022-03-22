import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

#PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
#st.beta_set_page_config(**PAGE_CONFIG)

def user_input_features() :
#   temp_amb = st.sidebar.slider('기온(°C)',0, 1, 0)
#   rainfall = st.sidebar.slider('강수량(mm)',0, 1, 0.5)
#   humid = st.sidebar.slider('습도',0, 1, 0.5)
#   height = st.sidebar.slider('고도(EL)',0, 1, 0.5)
  temp_amb = st.sidebar.slider('기온(°C)',-10, 15, 0)
  rainfall = st.sidebar.slider('강수량(mm)',0.0, 25.0, 0.05)
  humid = st.sidebar.slider('습도',0.0, 1.0, 0.2)
  height = st.sidebar.slider('고도(EL)',0.0, 50.0, 12.0)
  data = {'기온(°C)' : temp_amb,
          '강수량(mm)' : rainfall,
          '습도' : humid,
          '고도' : height
          }
  features = pd.DataFrame(data, index=[0])
  return features


def main():
	#st.title("Awesome Streamlit for MLDDD")
  st.write("""
  # Simple RF 동파예측 WebApp

  This app predicts the **Status of flowmeter** in winter!
  
  """)

  st.sidebar.header('User Input Parameters')

  df= user_input_features()

  st.subheader("주요 파라미터를 설정.")
  st.write(df)

#   iris = datasets.load_iris()
#   x=iris.data
#   y=iris.target

###########################
  df_read = pd.read_csv("./pipe_data.csv", encoding='cp949')
  print(df_read)
#   df_read.to_csv("pipe_data_return.csv", encoding='cp949')
###########################
 # x = df[['기온(°C)', '풍속(m/s)', '습도', '고도', '음/양지_0', '음/양지_1', '가정용/일반용_0', '가정용/일반용_1']]
  x = df_read[['기온(°C)', '강수량(mm)', '습도', '고도']]
  y = df_read['Grade']
#   print(x)
#   print(y)
  from sklearn.model_selection import train_test_split
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify=df_read['Grade'], random_state=42)

  #clf = RandomForestClassifier()
  clf = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, min_samples_split=2)

  clf.fit(x_train, y_train)

  predict_ = clf.predict(x_test)
  predict_proba = clf.predict_proba(x_test)

  st.subheader("예측 확률")
  st.write(predict_proba)

  st.bar_chart(predict_proba[3:10])

# -----------------------------------------------
  from sklearn.metrics import mean_absolute_error
  print('MAE score:', mean_absolute_error(y_test, predict_))
  st.subheader("함내 상태 예측오차")
  st.write(mean_absolute_error(y_test, predict_))  
# -----------------------------------------------

  st.subheader("Index (1)=Normal (2)=OK (3)=Serious (4)=Warning, (0)=Caution")
# print(df[' Grade'].value_counts())
# Normal     3052        1
# OK         2074          2
# Warning    1020        4
# Caution     882         0
# Serious     104         3

  st.line_chart(predict_)

  st.subheader("함내 상태 예측")
#   st.write(iris.target_names)
  st.write(y_test, predict_)

#   st.subheader("상태예측 2nd")
#   st.write(predict_)

#   st.subheader("상태예측 3rd")
#   st.write(iris.target_names)

if __name__ == '__main__':
	main()
