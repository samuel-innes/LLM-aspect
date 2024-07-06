"""
Visualise language-level entropy data from the TED2020 dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = {
    'Language': [
        'Arabic', 'Vietnamese', 'Chinese', 'Danish', 'Dutch', 'English', 'German',
        'Icelandic', 'Norwegian Bokmål', 'Norwegian Nynorsk', 'Swedish', 'Catalan',
        'French', 'Italian', 'Portugese', 'Romanian', 'Spanish', 'Finnish', 'Estonian',
        'Hungarian', 'Greek', 'Latvian', 'Lithuanian', 'Belarussian', 'Bulgarian',
        'Czech', 'Croatian', 'Polish', 'Russian', 'Slovakian', 'Slovene', 'Ukrainian'
    ],
    'Entropy': [
        0.2845309245, 0.4933570355, 0.27206972700000004, 0.430186455, 0.3601590548310915,
        0.316684746, 0.3482899846398908, 0.3946137650954421, 0.4636685439739413,
        0.42909772151898734, 0.4275604854418964, 0.4570681785, 0.410905858,
        0.3612088355, 0.41011952300000004, 0.33489905700000006, 0.316684746, 
        0.29564360085734703, 0.316684746, 0.3244019267833109, 0.26211523049999996,
        0.2865482285, 0.3482899846398908, 0.2845309245, 0.2845309245, 0.2751877900173932,
        0.2751877900173932, 0.30852190145653163, 0.28529353587227335, 0.2845309245,
        0.2845309245, 0.24490125931972792
    ],
    'Family': [
        'Semitic', 'Austroasiatic', 'Sino-Tibetan', 'Germanic', 'Germanic', 'Germanic', 'Germanic',
        'Germanic', 'Germanic', 'Germanic', 'Germanic', 'Romance', 'Romance', 'Romance',
        'Romance', 'Romance', 'Romance', 'Uralic', 'Uralic', 'Uralic', 'Hellenic',
        'Baltic', 'Baltic', 'Slavic', 'Slavic', 'Slavic', 'Slavic', 'Slavic', 'Slavic',
        'Slavic', 'Slavic', 'Slavic'
    ]
}


df = pd.DataFrame(data)

family_means = df.groupby('Family')['Entropy'].mean().reset_index()

family_means = family_means.sort_values('Entropy')

df = df.merge(family_means, on='Family', suffixes=('', '_mean'))

plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 13})
sns.barplot(y='Entropy', x='Family', data=family_means, color='lightgrey', edgecolor='black')

sns.stripplot(y='Entropy', x='Family', data=df, hue='Family', dodge=False, marker='o', alpha=0.9, size=10,jitter=False)

plt.title('Mean Language-Level Verbal Aspect Entropy by Family (TED2020)')
plt.ylabel('Entropy', fontweight='bold')
plt.xlabel('Language Family', fontweight='bold')
#plt.legend(title='Family', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Show plot
plt.savefig("/home/students/innes/ba2/LLM-aspect/img/lang_level_2020.jpeg", dpi=300)
plt.show()
