import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')


# Feature engineering
def preprocess(df, save_name):
    # Age
    df['Age'] = df['Age'].fillna(-1)
    df['Age'] = pd.cut(x=df['Age'], bins=[-2, 0, 3, 16, 35, 63, 150],
                       labels=['No age', 'Baby/Toddler', 'Child', 'Young Adult', 'Adult', 'Elderly'])

    # Ticket, Name, PassengerId
    df = df.drop(['Ticket', 'PassengerId', 'Name'], axis=1)

    # Cabin
    df['Cabin'] = df['Cabin'].fillna(0)
    df['Cabin'] = df['Cabin'].apply(lambda x: 0 if x == 0 else 1)

    # Add missing Fare
    median_3 = df.loc[(df['Pclass'] == 3)]['Fare'].median()
    df['Fare'] = df['Fare'].fillna(median_3)

    # Sanity check
    for column in df:
        assert df[column].isna().sum() == 0
    df.to_csv(f'data/{save_name}.csv')
    return df


df_train = preprocess(df_train, 'parsed_train')
y_train = df_train['Survived']
x_train = df_train.drop(['Survived'], axis=1)
x_train = x_train.select_dtypes(exclude=['number']).apply(LabelEncoder().fit_transform).join(x_train.select_dtypes(include=['number']))
x_test = preprocess(df_test, 'parsed_test')
x_test = x_test.select_dtypes(exclude=['number']).apply(LabelEncoder().fit_transform).join(x_test.select_dtypes(include=['number']))

model = LogisticRegression()
model.fit(x_train, y_train)
y_test = model.predict(x_test)
df_test['Survived'] = y_test
print(df_test)
predictions = df_test[['PassengerId', 'Survived']]
predictions.to_csv(f'data/predictions.csv')
