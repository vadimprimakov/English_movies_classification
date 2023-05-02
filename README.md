# Классификация фильмов по сложности восприятия английского языка

Запрос сформирован тем, что просмотр фильмов на оригинальном языке это популярный и действенный метод прокачаться при изучении иностранных языков. Важно выбрать фильм, который подходит студенту по уровню сложности, т.ч. студент понимал 50 - 70 % диалогов. Чтобы выполнить это условие, преподаватель должен посмотреть фильм и решить, какому уровню он соответствует. Однако это требует больших временных затрат от преподавателя.

О чем стоит подумать перед началом работы?
- датасет не очень большой, около 200 фильмов. Категорий много. Возможно нужно как-то расширить датасет. В Интернете есть фильмы с метками уровня сложности языка. Если каждый участник Мастерской найдет 10-20 дополнительных фильмов и после все объединят данные – получится уже хороший итоговый датасет, который можно использовать в работе!
- у некоторых фильмов указаны несколько меток, например А2/А2+. Нужно решить как быть с ними.

**Исходные данные**
Размеченный датасет с названиями фильмов, субтитрами и меткой уровня сложности языка (A1/A2/B1/B2/C1/C2)

**Полученные признаки**

- 'film_lenght', 
- 'text_len', 
- 'lematise_text_len', 
- 'sumb_persecond',
- 'phrases_lenght',
- 'sumb_persecond_frases',
- 'sumb_perword',
- 'num_sentence',
- 'word_persentence',
- 'A1' - доля уникальных слов сложности A1,
- 'A2' - доля уникальных слов сложности A2,
- 'B1' - доля уникальных слов сложности B1,
- 'B2' - доля уникальных слов сложности B2,
- 'C1' - доля уникальных слов сложности C1,
- 'C2' - доля уникальных слов сложности C2,
- 'phrases_count',
- 'words_unique_perphrase', 
- 'words_count',
- 'words_unique_count',
- 'words_unique_persecond',
- 'lexical_diversity',
- 'flesch_reading_ease',
- 'flesch_kincaid_grade',
- 'smog_index',
- 'coleman_liau_index' - уровень качества текста по формуле Коулмана-Лиау,
- 'automated_readability_index',
- 'dale_chall_readability_score',
- 'difficult_words',
- 'linsear_write_formula',
- 'gunning_fog',
- 'text_standard',
- 'fernandez_huerta',
- 'szigriszt_pazos',
- 'gutierrez_polini',
- 'crawford',
- 'gulpease_index',
- 'osman',
- 'num_idioms',
- gerund_persentence - среднее количество герундия во фразе


**Целевой признак**
- Level — уровень владения английским языком


## Цель исследования:

Определить необходимый уровень владения английским языком по предоставленному файлу с субтитрами к фильму или сериалу.

## Ход исследования:

Шаг 1. Открыть и загрузить исходные данные по ссылке: https://disk.yandex.ru/d/rQHuC6p6Ztf9Uw. Для обработки файла с субтитрами (формат srt) использована специализированная библиотека pysrt https://github.com/byroot/pysrt. <br/>

Шаг 2. Загрузка предварительно сформированных данных о сложности слов согласно Оксфордского словаря. В словарь добавлены размеченные слова уровня C2. <br/>

Шаг 3. Загрузка предварительно сформированного списка идиом. <br/>

Шаг 4. Проведени EDA. Для упрощения градаций преобразованы уровни сложности до минимального. <br/>

Шаг 5. Расширен датасет, получив информацию из открытых источников. В ходе поиска субтитров в датасет добавлены фильмы и сериалы категорий A1 и С2. <br/>

Шаг 6. Сформулированы и выбраны подходящие признаки (в том числе с помощью библиотеки NLTK и textstat).<br/>

Шаг 7. Подготовлены данные для обучения модели, выбрана метрика TotalF1.<br/>

Для решения поставленной задачи были применены разные модели машинного обучения, разные наборы признаков и разные подходы к кодированию текстовой информации. Я не буду приводить здесь все варианты решений, а остановлюсь только на том, который показал наибольшую эффективность:<br/>

Модель: CatBoostClassifier<br/>

Набор выбранных признаков:<br/>
- phrases_lenght - средняя длина фразы<br/>
- B2 - доля уникальных слов сложности B2<br/>
- coleman_liau_index - уровень качества текста по формуле Коулмана-Лиау<br/>
- word_persentence - среднее количество слов во фразе<br/>
- gulpease_index - индекс текста Gulpease<br/>
- gerund_persentence - среднее количество герундия во фразе<br/>
- words_unique_perphrase - среднее количество уникальных слов во фразе<br/>
- words_unique_persecond - среднее количество уникальных слов в секунду<br/>
- phrases_count - количество фраз в тексте<br/>
- avg_dificulty - средняя сложность слов<br/>
- idioms_persentence - среднее количество идиом во фразе<br/>

Кодировка текста производилась внутренним алгоритмом модели CatBoostClassifier. А в качестве текста выступали морфологические формы слов.<br/>

Шаг 8. Собрана baseline модель, с помощью gridsearch выполнен поиск наилучших гиперпараметров на основе кросс-валидации и оценено качество решения. <br/>
После множества попыток подбора гиперпараметров модели, остановился на следующих для CatBoostClassifier {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1, 'l2_leaf_reg': 1e-08}. Подбор гиперпараметров исключен из общего кода, чтобы немного разгрузить итоговых файл. Результат TotalF1 на 80% тренировочных данных - 0,68. Точность фиктивной модели примерно в два раза ниже, чем наша обученная модель.<br/>

Шаг 9. Наибольшую важность имеют признаки:<br/>

- words_unique_persecond - среднее количество уникальных слов в секунду<br/>
- words_unique_perphrase - среднее количество уникальных слов во фразе<br/>
- coleman_liau_index - уровень качества текста по формуле Коулмана-Лиау<br/>
- word_persentence - среднее количество слов во фразе<br/>


## Итог исследования:

На этапе подготовки данных была выполнена загрузка и первичный анализ данных, предобработка исходного датасета и формирования признакового пространства из информации находящийся в файлах субтитров. Для этого были использованы возможности библиотек pysrt, nltk и texstat. За рамками данной тетрадки остался процесс парсинга Оксфордского словаря сложности слов и составление словаря идиом. Также, файлы субтитров и сложность фильмов по классификации CEFR были внесены в датасет по результатам поиска в открытых источниках.

На этапе анализа данных были выявлены сильные корреляционные зависимости между большой частью признаков, но небыло выявлено корреляции признаков с целевой переменной. Это делало работу модели не простой задачей.

Для построения модели машинного обучения из большого числа вариантов кодирования, извлечения признаков и типов моделей машинного обучения был выбран наиболее эффективный. На тестовой выборке удалось достичь метрики качества TotalF1 = 0.67. Это в 2 раза лучше чем показывает фиктивная модель.

Факторы мешающие достижению большего качества, на мой взгляд, следующие:

Размер датасета очень небольшой. Добавлены субтитры в категориях A1 и С2, а также словарик для уровня C2.
Субъективность оценок уровня фильмов. В разных источниках один и тот же фильм мог иметь различный оценочный уровень. Отсюда и такие слабые различия между фильмами разных уровней. Сложность в разметке субтитров фильмов и сериалов делает трудозатратным увеличения выборки субтритров.
Увеличение качества классификации может быть достигнуто устранением вышеуказанных факторов.

Создано приложение на платформе стримлит, оно доступно по ссылке: https://english-subtitles-classification.streamlit.app/.


## Стек технологий:

`pandas`, `matplotlib`, `numpy`, `scikit-learn`, `seaborn`, `pysrt`, `nltk`, `textstat`, `CatBoostClassifier`, `GridSearchCV`

## Статус проекта:

Завершен.
