# bigduta
da


```
#импортирование библиотек
import numpy as np
import matploltlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
```
```
#1.1. загрузка данных 
df = pd.read_excel('path.xlsx')
df = pd.read_json('path.json')
```
```
#объединение дата фреймов
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html/df.drop('dropoff_datetime', axis=1, inplace=True) - дублирующиеся столбцы
#можно юзать всё вместе
df = pd.concat([df1, df2])
df = pd.drop__duplicates('column_name')
df = pd.drop('column_name', axis=1)
```
```
#приведение типов
#https://qastack.ru/programming/15891038/change-data-type-of-columns-in-pandas
df['column_name']=pd.to_datetime(df['column_name'])
```

```
#узнаем кол-во NaN-ов
df.isna().sum()
#выбросы и всякая штука
df.describe()

#перевод секунд в минуты, и после перевода опять сделать describe()
df.trip_duration = df.trip_duration.apply(lambda x: x/60)

#для более лучшего просмотра выбросов использовать boxplot
sns.boxplot(df['column_name'])
#очистка выбросов происходит за счёт подбора подходящих значения на boxplot
qunt = df['column_name'].quantile(0.9248)
df = df[df['column_name'] < qunt]
sns.boxplot(df['column_name'], palette = 'Blues_r')
plt.show()
```

```
# 1.2. df = pd.read_csv('name.csv'). 
name.info()
#просмотр данных, хранящихся в столбцах
for col in ['column_name', 'column_name', 'column_name']:
    print(weather[col].unique())
    print('-' * 10)
```

```
#замена неверных значений на "0" и конвертирование в другой тип данных
for col in ['precipitation', 'snow fall', 'snow depth']:
    weather.loc[weather[col] == 'T', col] = 0
    weather[col] = weather[col].astype(np.float64)
```

```
#просмотр выбросов
weather.describe()
#конвертирование из F в C
for temp in ['maximum temperature', 'minimum temperature','average temperature']:
    weather[temp] = weather[temp].apply(lambda x: (x-32)/1.8)
```

```
#Объединение поездок и погоды
#у нас в данных о поездках нету столбца date, мы добавим его временно, данные о годе, месяце и т.д. 
#возьмем из даты посадки. После объединения удалим столбец date и столбец с датой высадки
df['d'] = df['p.up_d.t.'].apply(lambda x: str(x.day) + '-' + str(x.month) + '-' + str(x.year))
df = df.merge(weather, on='d').drop(['d'], axis=1)
```

```
#добавление признаков кароче эту штуку надо сделать на каждый property
df['date_property(year\month\day\hour\minute)'] = df['dropoff_datetime'].apply(lambda x: x.year)
#dropoff_datetime можно удалить
```

```
#создание нового столбца для задания 
df['new_column']=df['old_column']
df.info()
```

```
# что такое квартиль? картинку чекни
# здесь короче у нас есть хрень с 0.25, потом нужно скопировать строку и написать 0.5, потом 0.75
# q2=df.category_travel_time.quantile(0.5) 
q1=df.category_travel_time.quantile(0.25)

# для q1 и q4 просто <= 0.25 и >= 0.75. То есть вот & юзать не надо
df.category_travel_time=np.where(((df.category_travel_time<q2)&(df.category_travel_time>q1)),2,df.category_travel_time)
# old_column можно удалить по причине его ненужности
```
```
#1.3 узнаем кол-во NaN-ов, плотность
new_csv.isna().sum()
df.info()
#для расшифровки и назначения чекни скрин
```
```
#1.3 узнаем кол-во NaN-ов, плотность
new_csv.isna().sum()
df.info()
#для расшифровки и назначения чекни скрин
```
```
#1.4 выбросы и всякая штука
df.describe()

#перевод секунд в минуты, и после перевода опять сделать describe()
df.trip_duration = df.trip_duration.apply(lambda x: x/60)

#для более лучшего просмотра выбросов использовать boxplot
sns.boxplot(df['column_name'])
```
```
#очистка выбросов происходит за счёт подбора подходящих значения на boxplot
qunt = df['column_name'].quantile(0.9248)
df = df[df['column_name'] < qunt]
sns.boxplot(df['column_name'], palette = 'Blues_r')
plt.show()
```
```
#1.5 сохраняет в директорию, где лежит Report_C1.ipynb
df.to_csv('C1_result.csv', index=False)
#Вывод: во время первой сессии были сделаны следующие вещи:
#1.обработаны пропуски во всех данных
#2.обработаны выбросы во всех данных
#3.данные приведены к приемлимому виду(отформатированы)
#4.объединены данные о всех поездках и о погоде в эти дни в один датасет и сохранены с расширением .csv
```
