import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

#PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
#st.beta_set_page_config(**PAGE_CONFIG)

def user_input_features() :
#   temp_amb = st.sidebar.slider('기온(°C)',0, -1, 0)
#   rainfall = st.sidebar.slider('강수량(mm)',0, 1, 0.5)
#   humid = st.sidebar.slider('습도',0, 1, 0.7)
#   height = st.sidebar.slider('고도(EL)',0, 1, 0.5)
  temp_amb = st.sidebar.slider('기온(°C)',-15, 15, -8)
  rainfall = st.sidebar.slider('강수량(mm)',0.0, 25.0, 8.5)
  humid = st.sidebar.slider('습도',0.0, 1.0, 0.15)
  height = st.sidebar.slider('고도(EL)',0.0, 50.0, 12.5)
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
  st.write("User 변수설정 : 좌측상단의 > 버튼 Click.")
  st.sidebar.header('User Input Parameters')

  df= user_input_features()

  st.subheader("[User의 주요 파라미터 설정치]")
  st.write(df)

###########################
  df_read = pd.read_csv("./pipe_data3r.csv", encoding='cp949')
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

########### START

  st.subheader("[사용자 입력에 따른 상태 예측]")
  st.write("Index : (0)=평상, (1)=관심, (2)=주의, (3)=경계, (4)=심각")
 
  predict_2 = clf.predict(df)	
  predict_proba2 = clf.predict_proba(df)
	
  st.write(predict_proba2)

#=========================================
  st.markdown(""" <style> .font {font-size:50px;} </style> """, unsafe_allow_html=True)
#=========================================
  st.write(" 동파안전 확률은  ",predict_proba2[0,0]*100, "% 입니다.")
  st.write(" 동파위험 확률은  ",100-predict_proba2[0,0]*100, "% 입니다.")
 # st.write(predict_2[0,0]) 
 
  st.bar_chart(predict_proba2)
  st.write(predict_2)

  

########### END
	
  st.subheader("[검증set 전체 예측 확률의 분포]")
  st.write(predict_proba)
  st.bar_chart(predict_proba[7:15])


# -----------------------------------------------
  from sklearn.metrics import mean_absolute_error
  print('MAE score:', mean_absolute_error(y_test, predict_))
  st.subheader("함내 상태 예측오차")
  st.write(mean_absolute_error(y_test, predict_))  
# -----------------------------------------------

  st.subheader("Index : (0)=Normal, (1)=OK, (2)=Caution, (3)=Warning, (4)=Serious")
# print(df[' Grade'].value_counts())
# Normal     3052        1
# OK         2074          2
# Warning    1020        4
# Caution     882         0
# Serious     104         3

  st.line_chart(predict_)

  st.subheader("[함내 상태 예측 범위 분포]")
#   st.write(iris.target_names)
  st.write(y_test, predict_)

#   chart_data = pd.DataFrame(
# 	  y_test, predict_,
# 	  columns=['실측치',''])
#   st.line_chart(chart_data)

#   st.subheader("상태예측 2nd")
#   st.write(predict_)

if st.checkbox('지도 표시(beta)'):
	map_data = pd.DataFrame(
		np.random.randn(10,2) / [50,50] + [35.74544992, 128.0814886],
    columns=['lat','lon'])
	st.map(map_data)

if __name__ == '__main__':
	main()

	

		
