#!/usr/bin/env python
# coding: utf-8

# ## 라이브러리와 데이터 로드

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

import datetime as dt
import missingno as msno
plt.rcParams['figure.dpi'] = 140


# In[4]:


get_ipython().system('pip install missingno')


# In[101]:


df = pd.read_csv('Desktop/netflix_titles.csv')
df


# ## 데이터 모양과 결측값 확인

# In[102]:


df.info()


# In[15]:


msno.matrix(df.sample(500))


# In[16]:


msno.bar(df)


# In[103]:


df.isna().sum()


# ## 결측값 채우기

# In[104]:


df['country'] = df['country'].fillna(df['country'].mode())


# In[105]:


df['cast'].replace(np.nan, 'Missing',inplace=True)
df['director'].replace(np.nan, 'Missing', inplace=True)


# In[106]:


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


# In[107]:


df.isnull().sum()


# ## 날짜시간 데이터 처리

# In[108]:


df['date_added'] = pd.to_datetime(df['date_added'])


# In[109]:


df.info()


# In[110]:


df['month_added'] = df['date_added'].dt.month
df['month_name_added'] = df['date_added'].dt.month_name()
df['year_added'] = df['date_added'].dt.year

df.head(3)


# ##  country  변수를 활용하여 파생변수 만들기

# In[111]:


df.tail(3)


# In[112]:


df['country_1st'] = df['country'].apply(lambda x : x.split(',')[0])
df['country_1st'].tail()


# In[113]:


df['country_1st'].replace('United States', 'USA', inplace=True)
df['country_1st'].replace('United Kingdom', 'UK', inplace=True)
df['country_1st'].replace('South Korea', 'S. Korea', inplace=True)
df['country_1st'].tail()


# ## rating  변수를 활용하여 파생변수 만들기

# In[114]:


df['rating'].unique()


# In[115]:


ratings_ages = {
    'TV-MA': 'Adults',
    'R': 'Adults',
    'PG-13':'Teens',
    'TV-14':'Teens',
    'TV-PG':'Older Kids',
    'NR': 'Adults',
    'TV-G': 'Kids',
    'TV-Y': 'Kids',
    'TV-Y7':'Older Kids',
    'PG':'Older Kids', 
    'G': 'Kids', 
    'NC-17': 'Adults', 
    'TV-Y7-FV': 'Older Kids', 
    'UR':'Adults'
}


# In[116]:


df['target_user'] = df['rating'].replace(ratings_ages)
df['target_user'].unique()


# ## duration  변수를 활용하여 파생변수 만들기

# In[117]:


df['duration'].unique()[:10]


# In[118]:


df['season_count'] = df.apply(lambda x: x['duration'].split(' ')[0] if 'Season' in 
x['duration'] else "", axis = 1)
df['season_count'].unique()


# In[119]:


df['season_count'].replace('',np.nan,inplace=True)


# In[120]:


df


# In[121]:


df['duration'] = df.apply(lambda x: x['duration'].split(' ')[0] if 'Season' not in
x['duration'] else "", axis =1)
df['duration'].unique()[:10]


# In[97]:


df


# In[138]:


df['duration'].replace('',np.nan, inplace=True)


# In[139]:


df


# ## listed_in 변숫값 분리하기

# In[140]:


df['genre'] = df['listed_in'].apply(lambda x : x.replace(' ,' , ',').replace(', ' , ',').split(','))


# In[141]:


df


# ## 불필요한 변수 삭제하기 ( 안지워짐..)

# In[142]:


df = df.drop(columns=['show_id', 'description'])
df.head(3)


# ## 데이터 타입 변경

# In[143]:


df.dtypes


# In[144]:


df['type'] = pd.Categorical(df['type'])
df['target_user'] = pd.Categorical(df['target_user'], categories=['Kids','Older Kids','Teens','Adults'])
df['year_added'] = pd.to_numeric(df['year_added'])
df['duration'] = pd.to_numeric(df['duration'])
df['season_count'] = pd.to_numeric(df['season_count'])


# In[145]:


df.dtypes


# In[146]:


df


# ## 콘텐츠 타입별로 데이터 저장

# In[147]:


df_tv = df[df['type'] == 'TV Show']
df_tv.head(3)


# In[148]:


df_movie = df[df['type'] == 'Movie']
df_movie.head(100)


# ## 데이터 시각화 기초

# In[149]:


import matplotlib.font_manager as fm

font_list = [font.name for font in fm.fontManager.ttflist]
font_list


# ##  콘텐츠 타입별 개수 시각화하기

# In[63]:


# 시본 그래프 스타일 지정: 한번만 실행
sns.set(style='white')

sns.countplot(x='type', data=df, palette='husl')
sns.despine(left=True)
plt.title('콘텐츠 타입별 개수', fontsize=14, fontfamily='Malgun Gothic',
         fontweight='bold', position=(0,0))

plt.show()


# ## 콘텐츠 타입 비중 시각화하기

# In[64]:


type_count = df.groupby(['type'])['type'].count()
length = len(df)
result = (type_count/length).round(2)


# In[65]:


(df.groupby(['type'])['type'].count()/length).round(2)


# In[66]:


type_ratio = pd.DataFrame(result).T
type_ratio


# In[68]:


labels = ['Movie', 'TV show']
wedgeprops = {'linewidth':2, 'width':1, 'edgecolor':'w'}

plt.figure(figsize=(6,4))
plt.pie(type_count/length, labels=labels, autopct='%1.2f%%', startangle=90,
colors = ['#E68193', '#459E97'],
       textprops={'fontsize': 10}, wedgeprops = wedgeprops)
plt.title('콘텐츠 타입 비중', fontsize=14, fontfamily='Malgun Gothic',
fontweight='bold', position=(0,0))
plt.show()


# ## 월별 콘텐츠 업로드 수 시각화 하기

# In[69]:


plt.figure(figsize=(10,5))
sns.countplot(x='month_added', hue='type', data=df, palette='husl')
sns.despine(left=True)
plt.title('월별 콘텐츠 업로드 수', fontsize=14, fontfamily='Malgun Gothic',
fontweight='bold', position=(0,0))

plt.show()


# ## 연도순으로 콘텐츠 개봉 또는 방영 수 시각화 하기

# In[72]:


plt.figure(figsize=(10,5))
sns.countplot(x='release_year', hue='type',data =df, palette='husl')
sns.despine(left=True)
plt.title('연도순 콘텐츠 개봉 또는 방영 수', fontsize=14,fontfamily='Malgun Gothic',
fontweight='bold', position=(0,0))
plt.xticks(rotation=90, fontsize=6)
plt.show() 


# ## 콘텐츠 타입별 타깃 유저 수 시각화 하기

# In[73]:


plt.figure(figsize=(10,5))
sns.countplot(x='target_user', hue='type', data = df, palette = 'husl')
sns.despine(left=True)
plt.title('콘텐츠 타입별 타깃 유저 수',fontsize=14,fontfamily='Malgun Gothic',
fontweight='bold', position=(0,0))

plt.xticks()
plt.show() 


# ## 연도순 콘텐츠 타입별 업로드 수 시각화하기

# In[74]:


plt.figure(figsize=(10,5))
sns.countplot(x='year_added', hue='type', data = df, palette = 'husl')
sns.despine(left=True)
plt.title('연도순 콘텐츠 타입별 업로드 수',fontsize=14,fontfamily='Malgun Gothic',
fontweight='bold', position=(0,0))

plt.xticks()
plt.show() 


# ## 연도순 콘텐츠 타입과 타깃 유저 비중 시각화하기

# In[77]:


sns.displot(x='year_added', hue='type', data=df, kind='kde', height=6,
           multiple='fill',clip=(0, None), palette='husl')
sns.despine(left=True)
plt.title('연도순 콘텐츠 타입 비중',fontsize=14,fontfamily='Malgun Gothic',
fontweight='bold', position=(0,0))


plt.xticks()
plt.show() 


# In[78]:


sns.displot(x='year_added',hue='target_user', data=df, kind='kde', height=6,
           multiple='fill',clip=(0, None), palette='husl')
sns.despine(left=True)
plt.title('연도순 타깃 유저 비중', fontsize=14, fontfamily='Malgun Gothic',
fontweight='bold', position=(0,0))


plt.xticks()
plt.show() 


# ## 영화와 TV쇼 장르 Top 10 시각화하기

# In[80]:


df_movie['listed_in'].value_counts().head(10)


# In[82]:


movie_top10 = df_movie['listed_in'].value_counts().head(10)

plt.figure(figsize=(15,5))
sns.barplot(x=movie_top10.index,
           y=movie_top10.values, palette='husl')
sns.despine(left=True)
plt.xticks(rotation=70)
plt.title('영화 Top 10 장르', fontsize=22, fontfamily='Malgun Gothic',
         fontweight='bold', position=(0,0))
plt.show()


# In[150]:


tv_top10 = df_tv['listed_in'].value_counts().head(10)

plt.figure(figsize=(15,5))
sns.barplot(x = tv_top10.index,
           y = tv_top10.values, palette = 'husl')
sns.despine(left=True)
plt.xticks(rotation=70)
plt.title('TV쇼 Top 10 장르', fontsize=22, fontfamily='Malgun Gothic',
         fontweight='bold', position=(0,0))

plt.show()


# ## 영화 재생 시간 분포와  TV쇼 시즌 수 시각화하기

# In[151]:


df_duration_over0 = df[df['duration'] > 0]


# In[152]:


plt.figure(figsize=(10,5))
sns.histplot(x='duration', bins=30, binwidth=4,kde=True, discrete=False,
            data=df_duration_over0)
sns.despine(left=True)
plt.title('영화 재생 시간 분포', fontsize=14, fontfamily='Malgun Gothic',
         fontweight = 'bold', position=(0,0))

plt.show()


# In[155]:


df_season_count = df['season_count'].value_counts().reset_index().sort_values('season_count', ascending=False)
df_season_count


# In[156]:


plt.figure(figsize=(10,5))
sns.barplot(x='index', y='season_count', data=df_season_count,palette='husl')
sns.despine(left=True)
plt.title('TV쇼 시즌 수', fontsize=14, fontfamily='Malgun Gothic',
         fontweight = 'bold', position=(0,0)) 
plt.xticks()
plt.show()


# ## 연도순 콘텐츠 업로드 수 시각화 하기

# In[157]:


contents_added_movie = df_movie['year_added'].value_counts().reset_index()
contents_added_movie = contents_added_movie.rename(columns = {'index': 'year_added', 'year_added': 'count'})
contents_added_movie = contents_added_movie.sort_values('year_added')
contents_added_movie['type'] = 'Movie'


# In[158]:


contents_added_movie


# In[159]:


contents_added_tv = df_tv['year_added'].value_counts().reset_index()
contents_added_tv = contents_added_tv.rename(columns = {'index': 'year_added', 'year_added':'count'})
contents_added_tv = contents_added_tv.sort_values('year_added')
contents_added_tv['type'] = 'TV show'


# In[162]:


contents_added_tv


# In[163]:


df_contents_added = pd.concat([contents_added_tv, contents_added_movie])
df_contents_added = df_contents_added.reset_index()
df_contents_added


# In[165]:


sns.relplot(x='year_added', y='count', hue='type', linewidth=2.5, palette='husl', kind='line', data=df_contents_added)
sns.despine(left=True)
plt.title('연도순 콘텐츠 타입별 업로드 수', fontsize=14, fontfamily='Malgun Gothic', fontweight='bold', position=(0,0))

plt.show


# ##  상관관계: 연도와 월 콘텐츠 업로드 수를 히트맵으로 시각화하기

# In[166]:


month_order = ['January', 'February', 'March', 'April','May', 'June', 'July', 'August','September','October', 'November','December'][::-1]
df_bymonth = df.groupby('year_added')['month_name_added'].value_counts().unstack().fillna(0)[month_order].T
df_bymonth


# In[167]:


plt.figure(figsize=(10,5))
sns.heatmap(df_bymonth, linewidths=.5, cmap='YlGnBu')
plt.title('연도와 월 콘텐츠 업로드 수 히트맵', fontsize=14, fontfamily='Malgun Gothic', fontweight='bold', position=(0,0))

plt.show()


# In[168]:


plt.figure(figsize=(10,5))
plt.pcolor(df_bymonth, cmap='afmhot_r',edgecolors='white',linewidths=2)

plt.title('연도와 월 콘텐츠 업로드 수 히트맵',fontsize=14, fontfamily='Malgun Gothic', fontweight='bold', position=(0,0)) 
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=8)
cbar.ax.minorticks_on()


# In[171]:


plt.xticks(np.arange(0.5, len(df_bymonth.columns),1), df_bymonth.columns,fontsize = 9, fontfamily='arial')
plt.yticks(np.arange(0.5, len(df_bymonth.index),1), df_bymonth.index,fontsize=9,fontfamily='arial')

plt.box(False)
plt.show()


# ## 랭킹: 콘텐츠 제작 국가 Top 10 시각화하기 

# In[176]:


top10_country = df.groupby('country_1st')['country'].agg('count').sort_values(ascending=False)[:10]
top10_country.head(10)


# In[179]:


# Top1 컬러만 다르게 하기 위한 컬러셋 변수 생성
colors = ['#f1f1f1' for _ in range(len(top10_country))]
colors[0] = '#E50914'


# In[180]:


plt.figure(figsize=(10,5))
plt.bar(top10_country.index, top10_country, width=0.8, linewidth=0.6, color=colors)
plt.grid(axis='y', linestype='-', alpha=0.2)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.title('Top 10 제작 국가',fontsize = 18, fontfamily='Malgun Gothic',
         fontweight='bold', position=(0,0))
sns.despine(top=True, right=True, left=True, bottom=False)
plt.show()


# In[181]:


top10_country_df = df.groupby('country_1st')['country'].agg('count').sort_values(ascending=False)[:10].reset_index()
top10_country_df.head(10)


# In[182]:


plt.figure(figsize=(10,5))
sns.barplot(x='country_1st', y='country', data=top10_country_df, palette='husl')
sns.despine(left=True)
plt.title('Top 10 제작 국가', fontsize=18, fontfamily='Malgun Gothic', fontweight='bold', position=(0,0))
plt.show()


# ## 워드 클라우드: 빈도가 높은 장르 시각화하기

# In[183]:


get_ipython().system('pip install wordcloud')


# In[184]:


from wordcloud import WordCloud, STOPWORDS


# In[185]:


text = ' '.join(df_movie['listed_in'])
text


# In[187]:


plt.figure(figsize=(10,5))
wordcloud = WordCloud(background_color='white', width=800, height=800, max_words=80, margin=10,random_state=1).generate(text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[ ]:




