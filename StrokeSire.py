import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

health_df=pd.read_csv('./healthcare-dataset-stroke-data.csv')
health_df


health_df.info()
# 데이터와 무관한 열 확인(id, 전처리 대상1)

sns.countplot(health_df['work_type'], label="Count")
plt.show()
# >> work_type 열에 Private 값이 많아서 없애야함 (전처리 대상 2)

health_df.isna().sum()
# >> NaN 값 확인 (bmi 열에 무려 201개, 전처리 대상 3)

sns.countplot(health_df['gender'], label="Count")
plt.show()
# >> gender 열에 Other 값이 포함되어 있다는 것을 확인 (전처리 대상 4)

sns.countplot(health_df['smoking_status'], label="Count")
plt.show()
# >> smoking_status 열에 Unknown 값이 포함되어 있다는 것을 확인 (전처리 대상 5)


# 전처리
# - 1) 무관한 id 열 삭제
# - 2) 무관한 work_type 열 삭제
# - 3) bmi == NaN 인 행에 bmi의 평균 값을 입력
health_df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/healthcare-dataset-stroke-data.csv')
health_df.drop(['id'], axis=1, inplace=True)
health_df.drop(['work_type'], axis=1, inplace=True)
health_df['bmi'].fillna(health_df['bmi'].mean(), inplace=True)
health_df.isna().sum() # 결측값 재확인

# - 4) (gender == Other) 행은 데이터 수가 더 적은 male 로 변경
# - 5) (smoking_status == Unknown) 행은 formerly smoked(이전에 핀적이 있음) 으로 변경 삭제
health_df.replace({'gender':'Other'}, 'Male', inplace=True)
health_df.replace({'smoking_status':'Unknown'}, 'formerly smoked', inplace=True)

health_df


# 인코딩
# - gender, ever_married, Residence_type, smoking_status 의 형을 string 에서 int 로 변환
# - age 행 값을 float >> int 형으로 변환
health_df.replace({'gender':'Male'}, 0, inplace=True)
health_df.replace({'gender':'Female'}, 1, inplace=True)

health_df.replace({'ever_married':'No'}, 0, inplace=True)
health_df.replace({'ever_married':'Yes'}, 1, inplace=True)

health_df.replace({'Residence_type':'Urban'}, 0, inplace=True)
health_df.replace({'Residence_type':'Rural'}, 1, inplace=True)

health_df.replace({'smoking_status':'never smoked'}, 0, inplace=True)
health_df.replace({'smoking_status':'formerly smoked'}, 0.5, inplace=True)
health_df.replace({'smoking_status':'smokes'}, 1, inplace=True)

health_df = health_df.astype({'age':int})
health_df


# 데이터 스케일링 - MinMaxScaler
# bmi 행 스케일링
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
bmi_data = health_df[['bmi']]
scaler.fit(bmi_data)
bmi_scale = scaler.transform(bmi_data)

print('bmi_data 조정 전 : \n {}'.format(bmi_data))
print('bmi_data 조정 후 : \n {}'.format(bmi_scale))

health_df['bmi'] = bmi_scale #정규화 한 값 대입

# avg_glucose_level 행 스케일링
scaler = MinMaxScaler()
avg_glucose_level_data = health_df[['avg_glucose_level']]
scaler.fit(avg_glucose_level_data)
avg_glucose_level_scale = scaler.transform(avg_glucose_level_data)

print('avg_glucose_level_data 조정 전 : \n {}'.format(avg_glucose_level_data))
print('avg_glucose_level_data 조정 후 : \n {}'.format(avg_glucose_level_scale))

health_df['avg_glucose_level'] = avg_glucose_level_scale #정규화 한 값 대입

# age 행 스케일링
scaler = MinMaxScaler()
age_data = health_df[['age']]
scaler.fit(age_data)
age_scale = scaler.transform(age_data)

print('age_data 조정 전 : \n {}'.format(age_data))
print('age_data 조정 후 : \n {}'.format(age_scale))

health_df['age'] = age_scale #정규화 한 값 대입


# 평가
from sklearn.model_selection import train_test_split

y_health_df = health_df['stroke']
X_health_df = health_df.drop('stroke', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_health_df, y_health_df, test_size=0.3, random_state=33)

# 모델 학습
# DecisionTreeClassifier(결정트리)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score  # 예측 성능 평가(정확도 평가)를 위한 API

dt_clf = DecisionTreeClassifier(random_state=33, max_depth=3) # 트리의 최대 깊이가 3일 때가 가장 정확도가 높음
dt_pred = dt_clf.fit(X_train, y_train).predict(X_test)
print('DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

# RandomForestClassifier(랜덤포레스트)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from numpy import sqrt

rf_clf = RandomForestClassifier(random_state=33, n_estimators=20, max_features=2)
rf_pred = rf_clf.fit(X_train, y_train).predict(X_test)
print('RandomForestClassifier 정확도:{0:.4f}'.format(accuracy_score(y_test, rf_pred)))

# Navie Bayes Classification(나이브 베이즈 분류)
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

nb_pred = nb_clf.fit(X_train, y_train).predict(X_test)
print('NavieBayesClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test, nb_pred)))

# 정확도 결과 시각화
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(3)
preds = [accuracy_score(y_test, dt_pred), accuracy_score(y_test, rf_pred), accuracy_score(y_test, nb_pred)]
models = ['DecisionTree', 'RandomForest', 'NavieBayes']

colors = ['#FF6B6B', '#F8CB2E', '#6BCB77']

plt.bar(x, preds, color = colors)
plt.xticks(x, models)
plt.ylim([0.85, 1])

plt.show()


# 평가
# K 폴드 교차 검증
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np

dt_clf = DecisionTreeClassifier()
kfold = KFold(n_splits=5)
cv_accuracy = []
n_iter = 0

print('K폴드 교차 검증 - 결정 트리\n')
for iter_count, (train_index, vd_index) in enumerate(kfold.split(X_train)):
    X_t, X_vd = X_train.values[train_index], X_train.values[vd_index]
    y_t, y_vd = y_train.values[train_index], y_train.values[vd_index]
    
    dt_clf.fit(X_train , y_train)    
    pred = dt_clf.predict(X_test)
    n_iter += 1

    # 반복 시 마다 정확도 측정 
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('#{0} 교차 검증 정확도 :{1: .4f}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'.format(n_iter, accuracy, train_size, test_size))
    
    cv_accuracy.append(accuracy)

print('\n## 평균 검증 정확도: {0: .4f}'.format(np.mean(cv_accuracy)))
