import streamlit as st
from srt_procesing import sub_processing
from catboost import CatBoostClassifier, Pool
import pandas as pd
import nltk
import pickle


print('Start app')
df_words = pd.read_csv('https://github.com/vadimprimakov/English_movies_classification/blob/main/app/oxford_dikt.csv', on_bad_lines='skip')
df_idioms = pd.read_csv('https://github.com/vadimprimakov/English_movies_classification/blob/main/app/theidioms_com.csv', sep='#', on_bad_lines='skip')
#model = CatBoostClassifier()
@st.experimental_memo
def load():
    with open('catboostclassifier_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
model = load()
#model.load_model('https://github.com/vadimprimakov/English_movies_classification/blob/main/app/catboostclassifier_model.pkl')
features = ['phrases_lenght', 
        'B2', 
        'coleman_liau_index', 
        'word_persentence', 
        'gulpease_index', 
        'gerund_persentence', 
        'words_unique_perphrase', 
        'words_unique_persecond',
        'phrases_count',
        'avg_dificulty',
        'idioms_persentence']

class f:
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    END = "\033[0m"

TITLE = 'Предсказание уровня английского языка по субтитрам'
st.set_page_config(
                   page_title=TITLE,
                   page_icon='🎬',
                   initial_sidebar_state='expanded',
                 )
st.title(TITLE)
st.write('Данное приложение на основе машинного обучения предсказыает уровень английского для фильмов и сериалов на английском языке. Все что нужно, это найти файлик с субтитрами, без этого никак. Сложность зависит от уровня по сертификации CEFR и может быть A1, A2, B1, B2, C1, C2.')

upload_file = st.file_uploader('Откройте файл субтитов в формате .srt', type='srt')

def make_predict(data, model):
    """
    :param data:
    :param model:
    :return:
    """
    predict_pool = Pool(data=data,
                       )
    predict = model.predict(predict_pool)
    decode = {1:'A1',
              2:'A2',
              3:'B1',
              4:'B2',
              5:'C1',
              6:'C2'
             }
    return decode[predict[0][0]]

def make_color(data, model):
    """
    :param data:
    :param model:
    :return:
    """
    predict_pool = Pool(data=data,
                       )
    predict = model.predict(predict_pool)
    if predict == 1 or 2:
        value_color = 'green'
    elif predict == 3 or 4:
        value_color = 'blue'
    elif predict == 5 or 6:
        value_color = 'red'
    return value_color

if upload_file:

    print(upload_file.name)

    df = sub_processing(upload_file, df_words, df_idioms)
    if df is None:
        st.write('Файл субтитров имеет неизвестный формат')
    else:
        st.header(f'Данный фильм имеет уровень **:{make_color(df[features], model)}[{make_predict(df[features], model)}]** по классификации CEFR')

        button = st.button('Показать анализ')
        if button:

            st.header('Содержание слов по сложности согласно Оксфордского словаря')
            st.bar_chart(df.loc[0, ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']])

            st.write(pd.DataFrame(
                {
                    'Характеристика':['Продолжительность фильма',
                                      'Количество слов',
                                      'Количество уникальных слов',
                                      'Количество фраз',
                                      'Средний темп речи',
                                      'Лексическое разнообразие',
                                     ],
                    'Значение':[f'{df.loc[0,"film_lenght"]//3600} ч. {(df.loc[0,"film_lenght"]%3600)//60} м.',
                                df.loc[0,'words_count'],
                                df.loc[0,'words_unique_count'],
                                df.loc[0,'phrases_count'],
                                f'{round(df.loc[0,"words_count"] / df.loc[0,"film_lenght"] * 60)} сл/мин',
                                round(df.loc[0,'lexical_diversity'], 3),
                               ],
                }
            ))

st.caption('------')


st.markdown("Больше проектов в профиле [GitHub](https://github.com/vadimprimakov)")
st.markdown("По остальным вопросам [Telegram](https://t.me/vadimprimakov)")