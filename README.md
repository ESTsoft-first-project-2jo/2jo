## 프로젝트 소개

### 서울시 아파트 매매가 예측
-----

최근, 부동산 가격이 급등락하며 가격예측이 힘들어지는 실정이며 아파트 부동산 매매의 타이밍에 대한 고민도 늘어가고 있다.

이에 정확한 아파트 가격을 예측하여 이런 이들에게 도움이 되고자 하는 마음에서 프로젝트를 시작하였다.

## 🕘 프로젝트 기간
**START  : 2024.02.21**
<br>
**END : 2024.03.07**

## 🧑‍💻 팀 구성
- **김세연** - 프로젝트 구상, EDA, 데이터 전처리, 모델링
- **이구협** - 조장, 발표, 프로젝트 구상, EDA, 데이터 전처리, 이상치 확인, 모델링, 앙상블, 하이퍼파라미터 최적화, GUI 구현
- **정우성** - 프로젝트 구상, EDA, 데이터 전처리, 이상치 확인, 모델링, 앙상블, 하이퍼파라미터 최적화
- **최한솔** - 프로젝트 구상, 발표 자료 준비, EDA, 데이터 전처리, 모델링

## ⌨ 개발 환경
### Language
------
<img src="https://img.shields.io/badge/language-%23121011?style=for-the-badge"><img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"><img src="https://img.shields.io/badge/3.11.9-515151?style=for-the-badge"><br>


### IDE 
------
<img src="https://img.shields.io/badge/ide-%23121011?style=for-the-badge"><img src="https://img.shields.io/badge/visual studio code-007ACC?style=for-the-badge&logo=visual studio code&logoColor=white"><br>
<img src="https://img.shields.io/badge/ide-%23121011?style=for-the-badge"><img src="https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white"> 


## 전체적인 모델링 프로세스
### 데이터 분석 및 EDA, 전처리
------

#### 데이터 추출

[서울시 부동산 실거래가 정보] (https://data.seoul.go.kr/dataList/OA-21275/S/1/datasetView.do)

> 해당 데이터셋에서 2020년 1월 1일부터 2024년 2월 28일까지 거래된 데이터를 사용하였음.

> 원본 데이터에서는 1,332,702개의 Row가 존재했으며, 이를 2020년 이후 데이터로 잘라내어 약 180,000개의 데이터만을 사용함

- 자치구명
- 법정동명
- 본번
- 부번
- 건물명
- 계약일
- 물건금액
- 건물면적
- 층
- 건축년도


#### Geocoding

> 네이버 지도 API를 사용하여 원본 데이터에 존재하는 '본번', '부번'을 이용해 지번주소를 만든 뒤, 지번 주소를 지오코딩하여 위도, 경도로 변환함.

#### 추가적인 데이터

- 서울시 병원의 위치와 정보
- 서울시 학교의 위치와 정보
- 서울시 지하철역의 위치와 정보
- 서울시 버스 정류소의 위치와 정보
- 서울시 편의시설(상업시설, 병의원, 공원 등)의 위치와 정보
- 전세가격지수
- 소비자 물가지수
- 월별 기준 금리
- 서울시 법정동별 나이별 인구 자료
- 아파트 매매가 실거래 가격 지수
- etc.

#### 인구 정보 변경

```
population['어린이인구'] = population[[f"{age}세남자" for age in range(13)] + [f"{age}세여자" for age in range(13)]].sum(axis=1)
population['청소년인구'] = population[[f"{age}세남자" for age in range(13, 25)] + [f"{age}세여자" for age in range(13, 25)]].sum(axis=1)
population['청년인구'] = population[[f"{age}세남자" for age in range(25, 41)] + [f"{age}세여자" for age in range(25, 41)]].sum(axis=1)
population['중장년인구'] = population[[f"{age}세남자" for age in range(41, 66)] + [f"{age}세여자" for age in range(41, 66)]].sum(axis=1)
population['노년인구'] = population[[f"{age}세남자" for age in range(66, 109)] + [f"{age}세여자" for age in range(66, 109)]].sum(axis=1)
```

> 0세부터 109세까지 1세씩 정리되어있던 나이를 어린이 (0~12), 청소년(13~24), 청년(25~40), 중장년(41~65), 노년(66~109)으로 나눔

#### 타겟변수 변경

<img width="486" alt="image" src="https://github.com/ESTsoft-first-project-2jo/2jo/assets/160453988/ca630e5d-f50e-4bfd-af61-ae8f825517f2">

> 아파트 매매가 데이터에 있던 "매매가"를 "면적당 가격"으로 변경


#### 한강 데이터 활용

```
from scipy.interpolate import interp1d

hangang_sorted = bridge.sort_values(by='Longitude')

# 다리를 찍을 좌표를 선형보간으로 잇습니다.
interpolate_lon = np.linspace(hangang_sorted['Longitude'].min(), hangang_sorted['Longitude'].max(), 130) #선 상에 있는 좌표를 기록합니다. 130개
linear_interp = interp1d(hangang_sorted['Longitude'], hangang_sorted['Latitude'], kind='linear')
interpolate_lat = linear_interp(interpolate_lon)

selected_coords = np.column_stack((interpolate_lon, interpolate_lat)) 

# 선택된 좌표를 담은 데이터프레임 생성
selected_coords_df = pd.DataFrame(selected_coords, columns=['Longitude', 'Latitude'])
```

> 선울 한강을 지나는 교량의 중심좌표를 선형보간하여 좌표를 추출한뒤, 아파트와의 거리를 Haversine공식을 이용하여 측정하였음.

#### 이상치, VIF 검사, P-value확인

```
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X.astype(float)
X_const = add_constant(X) 

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
vif["features"] = X_const.columns

vif["VIF Factor"] = vif["VIF Factor"].apply(lambda x: '{:.0f}'.format(x))
vif
```

> 이상치는 있었지만, 실제 데이터이므로 이상치에 대한 제거는 생략하였으며, 다중공선성검사를 통해 VIF값이 높은 변수를 제거하였음. 제거하고나니 P-value는 낮게나옴

#### 후진소거법, 전진소거법, 교차선택법

```
#전진선택법

def forward_selection(data, target, significance_level=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    aic_values = []
    
    while len(initial_features) > 0:
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < significance_level:
            best_feature = new_pval.idxmin()
            best_features.append(best_feature)
            aic_values.append(sm.OLS(target, sm.add_constant(data[best_features])).fit().aic)
        else:
            break
    
    return best_features, aic_values

#후진소거법
def backward_elimination(data, target, significance_level = 0.05):
    features = data.columns.tolist()
    while len(features) > 0:
        features_with_constant = sm.add_constant(data[features])
        p_values = sm.OLS(target, features_with_constant).fit().pvalues[1:] 
        max_p_value = p_values.max()
        if max_p_value >= significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break 
    return features

#교차선택법
def stepwise_selection(data, target, SL_in=0.05, SL_out=0.05):
    initial_features = data.columns.tolist()
    best_features = []
    while len(initial_features) > 0:
        changed=False
        # 전진 선택
        remaining_features = list(set(initial_features) - set(best_features))
        new_pval = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            model = sm.OLS(target, sm.add_constant(data[best_features + [new_column]])).fit()
            new_pval[new_column] = model.pvalues[new_column]
        min_p_value = new_pval.min()
        if min_p_value < SL_in:
            best_features.append(new_pval.idxmin())
            changed=True
        
        # 후진 소거
        model = sm.OLS(target, sm.add_constant(data[best_features])).fit()
        p_values = model.pvalues.iloc[1:]
        max_p_value = p_values.max()
        if max_p_value > SL_out:
            changed=True
            worst_feature = p_values.idxmax()
            best_features.remove(worst_feature)
        
        if not changed:
            break

    return best_features

```

> 이미 VIF 검사를 통해 변수를 제거해서, 유의미한 결과를 얻지 못함.



### 모델링

#### 최종결과

|모델|R^2|MSE|RMSE|MAE|
|----|---|---|---|---|
|RF|0.9249|0.0158|0.1256|0.0862|
|CatBoost|0.9232|0.0161|0.1270|0.0927|
|KNN 회귀|0.8915|0.0228|0.1510|0.1032|
|MLP 회귀|0.9045|0.0201|0.1417|0.1052|
|XGBoost|0.9293|0.0148|0.1218|0.0879|
|스태킹 앙상블|0.9273|0.0153|0.1235|0.0896|

#### 사용한 Columns

- 위도, 경도, 상위 10개 건설사 여부, 연식, 층수, 기준금리, 매매가 대비 전세가
- 법정동 전체 인구, 법정동 어린이, 청년, 노년 비율
- 자치구별 연간 소비액, 가장 가까운 종합병원과의 거리, 가장 가까운 지하철역과의 거리, 0.7km내 초등학교 개수, 2km내 명문 일반고 개수
- 1km내 일반 병의원과 상업시설의 개수, 0.8km내 공원 및 하천의 존재 여부
- 한강변에서 0.4km이내에 있는지 여부, 0.5km내 버스 정류장 수의 절반, 라벨 인코딩 된 자치구

#### 학습 데이터 분할

> 2024년 1월 1일 ~ 2024년 2월 28일 데이터를 Test Set으로 만들고, 전체 데이터를 8:2로 Train/Val로 나눔

#### 스케일링

> 스케일링이 필요한 KNN, 선형회귀만 스케일링을 적용하였음

#### 하이퍼파라미터 튜닝

> Optuna 라이브러리를 이용하여 최적화하였음

#### 앙상블

```
base_models = [
    ('Random Forest', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ('CatBoost', CatBoostRegressor(iterations=299, depth=10, learning_rate=0.2953026, random_strength=7, bagging_temperature=0.02308666, border_count=130, l2_leaf_reg=0.042969, random_state=42)),
    ('KNN', knn_pipeline),
    ('Decision Tree', DecisionTreeRegressor(random_state=42))
]

meta_model = XGBRegressor(n_estimators=563, max_depth=3, learning_rate=0.0151, min_child_weight=1, subsample=0.8967, colsample_bytree=0.9831, reg_alpha=3.029,reg_lambda=0.9275, random_state=42)

stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

stacking_model.fit(X_train, y_train)
```

> 성능이 좋은 모델들을 스태킹 하였음


### 결과

<img width="389" alt="image" src="https://github.com/ESTsoft-first-project-2jo/2jo/assets/160453988/42a79383-ab11-45c8-bcad-ab977d45374a">

> 실제로 Test셋에 가장 적합했던건 XGBoost 였음.



## 추가로, 실제 데이터를 통한 예측

### GUI

#### GUI 설계

```
def update_neighborhoods(*args):
    selected_district = district_var.get()
    neighborhoods_menu['menu'].delete(0, 'end')
    for neighborhood in districts[selected_district]:
        neighborhoods_menu['menu'].add_command(label=neighborhood, command=lambda n=neighborhood: neighborhood_var.set(n))
    neighborhood_var.set(list(districts[selected_district])[0])

def search_action():
    search_input = neighborhood_var.get() + ' ' + entry.get()
    result_text.delete(1.0, "end") 
    result_text.insert("end", "검색어: " + search_input + "\n")
    print("검색어:", search_input)
    search_url = f"https://m.land.naver.com/search/result/{search_input}#mapFullList"

    driver.get(search_url)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    script = soup.find('script', string=re.compile(r'lat\s*:\s*\'\d+\.\d+\''))
    if script:
        lat_match = re.search(r'lat\s*:\s*\'(\d+\.\d+)\'', script.string)
        lng_match = re.search(r'lng\s*:\s*\'(\d+\.\d+)\'', script.string)
        if lat_match and lng_match:
            lat = lat_match.group(1)
            lng = lng_match.group(1)
            predicted_price = predict_price(lat, lng, search_input, district_var.get(), neighborhood_var.get())
            result_text.insert("end", f"예측된 가격: {int(predicted_price[0])}만원\n")

        else:
            result_text.insert("end", "위도와 경도를 찾을 수 없습니다.\n")
    else:
        print("위도와 경도 정보를 포함하는 스크립트를 찾을 수 없습니다.")
    result_text.pack() 
    

root = tk.Tk()
root.title("네이버 부동산 검색")

result_text = tk.Text(root, height=10, width=50)

district_var = tk.StringVar(root)
district_var.set('선택하세요') 
district_var.trace('w', update_neighborhoods) 
districts_menu = ttk.OptionMenu(root, district_var, *districts.keys())
districts_menu.pack(side=tk.LEFT, padx=10)

neighborhood_var = tk.StringVar(root)
neighborhoods_menu = ttk.OptionMenu(root, neighborhood_var, '')
neighborhoods_menu.pack(side=tk.LEFT, padx=10)
update_neighborhoods()

entry = ttk.Entry(root)
entry.pack(side=tk.LEFT, padx=10)

search_button = ttk.Button(root, text="검색", command=search_action)
search_button.pack(side=tk.LEFT, padx=10)

root.mainloop()
```

> tkinter와 selenium으로 구현하였음

#### 대략적인 구조
<img width="376" alt="image" src="https://github.com/ESTsoft-first-project-2jo/2jo/assets/160453988/c780eb75-8039-47a3-9c75-a3e746c84400">

> 기본 화면

<img width="376" alt="image" src="https://github.com/ESTsoft-first-project-2jo/2jo/assets/160453988/5f83c8f9-453f-4b21-b123-8ca33fa274df">

> 구와 동을 선택하고 아파트 가격을 입력하면 기존에 학습된 머신러닝 모델을 통해서 가격예측을 시작함


<img width="599" alt="image" src="https://github.com/ESTsoft-first-project-2jo/2jo/assets/160453988/679f8672-2bca-468d-a421-1432d89b40ad">

> 실제 데이터와 큰 차이가 없는 모습을 볼 수 있다.



