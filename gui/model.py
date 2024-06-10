


def predict_price(lat, lng, name, gu, dong):

    import pandas as pd
    import numpy as np
    from scipy.interpolate import interp1d
    import joblib

    loaded_model = joblib.load('price_model_gui.pkl')

    Population = pd.read_csv('Population.csv', encoding='cp949')
    consumption = pd.read_csv('consumption.csv',encoding='utf-8')
    station = pd.read_csv('station.csv', encoding='utf-8')
    school = pd.read_csv('school.csv', encoding='utf-8')
    hospital = pd.read_csv('Hospital.csv', encoding='cp949')
    facility = pd.read_csv('facility.csv', encoding='cp949')
    bridge = pd.read_csv('bridge.csv', encoding='utf-8')

    medical_center = hospital[hospital['병원분류명'] == '종합병원']
    hospital = hospital[hospital['병원분류명'] != '종합병원']
    elementary_school = school[school['학교종류명'] == '초등학교']
    school = school[school['일반명문고'] == 'o']

    Mall = facility[facility['시설용도분류'] == 'FU_BA']
    Hospital = facility[(facility['시설용도분류'] == 'FU_BD') | (facility['시설용도분류'] == 'FU_BE')| (facility['시설용도분류'] == 'FU_BH')| (facility['시설용도분류'] == 'FU_BG')]
    Park = facility[(facility['시설용도분류'] == 'FU_BI') | (facility['시설용도분류'] == 'FU_BJ')]

    consumption.drop(0, inplace=True)
    consumption.drop(columns='경제활동별(1)', inplace=True)
    consumption.rename(columns={"자치구(1)":'Gu', '2021':'Consumption'}, inplace=True)
    
    df = pd.DataFrame()
    df.at[0, 'name'] = name
    df.at[0, 'Gu'] = gu
    df.at[0, 'dong'] = dong
    df.at[0, 'Longitude'] = lng
    df.at[0, 'Latitude'] = lat
    df.at[0, 'Interest Rate'] = 3.5
    df.at[0, 'Jeonse Index'] = 54.2
    
    df = pd.merge(df, consumption, how='left', on='Gu')

    df['Consumption'] = df['Consumption'].astype(float)/10000
    df['Consumption'] = np.log1p(df['Consumption'])

    major2 = ['자이','힐스테이트','디에이치','현대','편한세상','아크로','더샵','오티에르','래미안','삼성','푸르지오']
    #Top 10 건설사 : 롯데건설, SK,한화,HDC
    major1 = ['롯데','르엘','아이파크','드파인','아펠바움','SK','호반','포레나','꿈에그린','오벨리스크']
    df['major']=0 #0으로 초기화
    for i in range(len(df)):
        if any(substring in df.loc[i, 'name'] for substring in major1):
            df.loc[i, 'major'] = 1
            print(1)
        if any(substring in df.loc[i, 'name'] for substring in major2):
            df.loc[i, 'major'] = 2


    hangang_sorted = bridge.sort_values(by='Longitude')

    # 다리를 찍을 좌표를 선형보간으로 잇습니다.
    interpolate_lon = np.linspace(hangang_sorted['Longitude'].min(), hangang_sorted['Longitude'].max(), 130) #선 상에 있는 좌표를 기록합니다. 130개
    linear_interp = interp1d(hangang_sorted['Longitude'], hangang_sorted['Latitude'], kind='linear')
    interpolate_lat = linear_interp(interpolate_lon)

    selected_coords = np.column_stack((interpolate_lon, interpolate_lat)) 

    # 선택된 좌표를 담은 데이터프레임 생성
    selected_coords_df = pd.DataFrame(selected_coords, columns=['Longitude', 'Latitude'])


    def haversine(lat1, lon1, lat2, lon2): #Haversine 함수를 정의합니다. 
        
        #속도를 빠르게하기 위해 데이터프레임을 직접 사용하지않고 넘파이 어레이를 사용했습니다.
        lat1, lon1, lat2, lon2 = map(np.radians, [np.array(x).astype(float) for x in [lat1, lon1, lat2, lon2]]) 
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = 6371 * c # 6371 = 지구 반지름
        
        return distance

    #가장 가까운 역의 이름과 그 거리를 기록하는 함수입니다.

    def find_nearest_stations(df, station):

        #마찬가지로 넘파이 어레이로 변형합니다.
        lat1 = df['Latitude'].values[:, np.newaxis]
        lon1 = df['Longitude'].values[:, np.newaxis]
        lat2 = station['Latitude'].values
        lon2 = station['Longitude'].values

        distances = haversine(lat1, lon1, lat2, lon2)
        nearest_idx = np.argmin(distances, axis=1)
        nearest_stations = station.iloc[nearest_idx]
        
        # 가장 가까운 역의 이름과 거리를 DataFrame에 추가합니다.
        df['Nearest Station'] = nearest_stations['Station Name'].values
        df['Distance to NS'] = np.min(distances, axis=1)


    #가장 가까운 종합병원의 이름과 그 거리를 기록하는 함수입니다.

    def find_nearest_medical_centers(df, medical_center):

        lat1 = df['Latitude'].values[:, np.newaxis]
        lon1 = df['Longitude'].values[:, np.newaxis]
        lat2 = medical_center['Latitude'].values
        lon2 = medical_center['Longitude'].values

        distances = haversine(lat1, lon1, lat2, lon2)
        nearest_idx = np.argmin(distances, axis=1)
        nearest_medical_centers = medical_center.iloc[nearest_idx]
        
        df['Nearest Medical Center'] = nearest_medical_centers['기관명'].values
        df['Distance to MC'] = np.min(distances, axis=1)


    find_nearest_medical_centers(df, medical_center)
    find_nearest_stations(df, station)


    def count_schools(apartment, school, km1):
        
        counts = []

        lat1 = apartment['Latitude'].values[:, np.newaxis]
        lon1 = apartment['Longitude'].values[:, np.newaxis]
        lat2 = school['Latitude'].values
        lon2 = school['Longitude'].values

        for i in range(len(lat1)):
            distances = haversine(lat1[i], lon1[i], lat2, lon2)

            count = np.sum(distances <= km1) #거리를 수정
            counts.append(count)

        return counts

    #거리 안에 있는 고등학교를 세는 함수. 명문고등학교만을 포함합니다. 명문 고등학교는 일반고등학교 중에서 서울대 진학률 상위 50개 학교만 기록하엿습니다.

    def count_high_schools(apartment, school, km2):
        
        counts = []

        lat1 = apartment['Latitude'].values[:, np.newaxis]
        lon1 = apartment['Longitude'].values[:, np.newaxis]
        lat2 = school['Latitude'].values
        lon2 = school['Longitude'].values

        for i in range(len(lat1)):
            distances = haversine(lat1[i], lon1[i], lat2, lon2)

            count = np.sum(distances <= km2 ) #거리를 수정
            counts.append(count)

        return counts

    #거리 안에 있는 상업시설을 세는 함수

    def count_market(apartment, market, km3):
        
        counts = []

        lat1 = apartment['Latitude'].values[:, np.newaxis]
        lon1 = apartment['Longitude'].values[:, np.newaxis]
        lat2 = market['Latitude'].values
        lon2 = market['Longitude'].values

        for i in range(len(lat1)):
            distances = haversine(lat1[i], lon1[i], lat2, lon2)
            count = np.sum(distances <= km3)
            counts.append(count)

        return counts

    #거리 안에 있는 일반 병의원 수를 세는  함수

    def count_hostipal(apartment, hostipal, km4):
        
        counts = []

        lat1 = apartment['Latitude'].values[:, np.newaxis]
        lon1 = apartment['Longitude'].values[:, np.newaxis]
        lat2 = hostipal['Latitude'].values
        lon2 = hostipal['Longitude'].values

        for i in range(len(lat1)):
            distances = haversine(lat1[i], lon1[i], lat2, lon2)
            count = np.sum(distances <= km4)
            counts.append(count)

        return counts

    #거리안에 공원, 하천이 있다면 1을, 없다면 0을 반환하는 함수

    def find_park(apartment, park, km5):
        
        counts = []

        lat1 = apartment['Latitude'].values[:, np.newaxis]
        lon1 = apartment['Longitude'].values[:, np.newaxis]
        lat2 = park['Latitude'].values
        lon2 = park['Longitude'].values

        for i in range(len(lat1)):
            distances = haversine(lat1[i], lon1[i], lat2, lon2)
            count = np.any(distances <= km5).astype(int)
            counts.append(count)

        return counts

    #한강 중심으로 부터 정해진 거리만큼 떨어진 거리안에 있다면 1을, 아니면 0을 반환하는 함수

    def find_hangang(apartment, hangang, km6):
        
        counts = []

        lat1 = apartment['Latitude'].values[:, np.newaxis]
        lon1 = apartment['Longitude'].values[:, np.newaxis]
        lat2 = hangang['Latitude'].values
        lon2 = hangang['Longitude'].values

        for i in range(len(lat1)):
            distances = haversine(lat1[i], lon1[i], lat2, lon2)
            count = np.any(distances <= km6).astype(int)
            counts.append(count)

        return counts

    #km 수치는 지도를 확인하면서, 수치를 바꿔가면서 확인하였음

    df['Elementary Schools Num'] = count_schools(df, elementary_school, 0.7)
    df['High Schools Num'] = count_high_schools(df, school, 2)
    df['Market Num'] = count_market(df, Mall, 1)
    df['Hospital Num'] = count_hostipal(df, Hospital, 1)
    df['Park Presence'] = find_park(df, Park, 0.8)
    df['Nearby Hangang'] = find_hangang(df, selected_coords_df, 0.9)



    population = Population[Population['시도명'] == '서울특별시'].copy()

    population['어린이인구'] = population[[f"{age}세남자" for age in range(13)] + [f"{age}세여자" for age in range(13)]].sum(axis=1)
    population['청소년인구'] = population[[f"{age}세남자" for age in range(13, 25)] + [f"{age}세여자" for age in range(13, 25)]].sum(axis=1)
    population['청년인구'] = population[[f"{age}세남자" for age in range(25, 41)] + [f"{age}세여자" for age in range(25, 41)]].sum(axis=1)
    population['중장년인구'] = population[[f"{age}세남자" for age in range(41, 66)] + [f"{age}세여자" for age in range(41, 66)]].sum(axis=1)
    population['노년인구'] = population[[f"{age}세남자" for age in range(66, 109)] + [f"{age}세여자" for age in range(66, 109)]].sum(axis=1)
    population = population[['읍면동명', '계', '어린이인구', '청소년인구', '청년인구', '중장년인구', '노년인구']]
    population2 = population.rename(columns={'읍면동명':'dong', '계':'Total Population', '어린이인구':'Children', '청소년인구':'Adolescent', '청년인구':'Youth', '중장년인구':'Middle-Aged', '노년인구':'Old Age'})

    merged_df = pd.merge(df, population2, on='dong', how='left')

    merged_df['Children'] = merged_df['Children']/merged_df['Total Population']
    merged_df['Youth'] = merged_df['Youth']/merged_df['Total Population']
    merged_df['Adolescent'] = merged_df['Adolescent']/merged_df['Total Population']
    merged_df['Middle-Aged'] = merged_df['Middle-Aged']/merged_df['Total Population']
    merged_df['Old Age'] = merged_df['Old Age']/merged_df['Total Population']

    merged_df['Longitude'] = merged_df['Longitude'].astype(float)
    merged_df['Latitude'] = merged_df['Latitude'].astype(float)

    df2 = merged_df[['Longitude',
       'Latitude', 'major','Interest Rate', 'Jeonse Index', 'Total Population',
       'Children', 'Youth', 'Old Age',
       'Consumption', 'Distance to MC', 'Distance to NS', 'Elementary Schools Num',
       'High Schools Num', 'Market Num', 'Hospital Num', 'Park Presence',
       'Nearby Hangang']]

    pred_xgb = loaded_model.predict(df2)

    return pred_xgb



