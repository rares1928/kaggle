import pandas as pd

data = pd.read_csv('data/train.csv')
print(data['Ticket'].isna().sum())

# Idei pentru Age:
# 1) facem categorii: no_age, infant, child, young_adult, adult, middle_age, almost_dead
# 2) adaugam o coloana cu have_age si lasam si null ca valoare posibila pt Age
# 3) punem un default value (mediana, 0, infinit)

# Idei pentru Ticket:
# 1) o ignoram
# 2) split dupa space + pastram ce e inainte de spatiu fara puncte si slash-uri si facut niste categorii
#
# Idei pentru Cabin:
# 1) are cabina n-are cabina
# 2) grupam pe litere