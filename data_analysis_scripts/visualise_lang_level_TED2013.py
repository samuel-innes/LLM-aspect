"""
Visualise language-level entropy data from the TED2013 dataset.
"""

import matplotlib.pyplot as plt
import numpy as np

languages = [
    'Mandarin', 
    'English', 
    'German', 
    'Dutch',
    'Italian', 
    'Romanian', 
    'Spanish', 
    'Portugese',
    'Polish', 
    'Slovene', 
    'Russian'
]

mean_entropies = [
    0.2457926865, 
    0.32071384950000004, 
    0.35084973456589896, 
    0.369846907,
    0.3622703105, 
    0.3364678925, 
    0.361938744, 
    0.412085471,
    0.3117971745038716, 
    0.2982481445466491, 
    0.28983730150000003
]

variances = [
    0.3364717757426732, 0.19311976839891637, 0.4213040355252157, 0.41680491698461686,
    0.42906487671743604, 0.40408289302553185, 0.4318405802806199, 0.4371761285688277,
    0.4116000590646025, 0.38112935659527475, 0.3874103582350285
]

# sort data by mean_entropies
sorted_indices = np.argsort(mean_entropies)
languages_sorted = [languages[i] for i in sorted_indices]
mean_entropies_sorted = [mean_entropies[i] for i in sorted_indices]
variances_sorted = [variances[i] for i in sorted_indices]

colors = []
cm_colors = plt.cm.tab10(np.linspace(0, 1, len(mean_entropies)))

for lang in languages_sorted:
    if lang in ['Russian', 'Polish', 'Slovene']:
        colors.append(cm_colors[0])
    elif lang in ['French', 'Spanish', 'Romanian', 'Portugese', 'Italian']:
        colors.append(cm_colors[1])
    elif lang in ['German', 'Dutch', 'English']:
        colors.append(cm_colors[2])
    elif lang == 'Mandarin':
        colors.append(cm_colors[3])
    else:
        colors.append('gray')  # Default color if not in any group

x = np.arange(len(languages_sorted))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
plt.rcParams.update({'font.size':15})


bars1 = ax.bar(x, mean_entropies_sorted, width, color=colors)

ax.set_xlabel('Language', fontsize=13, fontweight='bold')
ax.set_ylabel('Entropy', fontsize=13, fontweight='bold')
ax.set_title('Mean Language-Level Verbal Aspect Entropy of Different Languages (TED2013)')
ax.set_xticks(x)
ax.set_xticklabels(languages_sorted, fontsize=13)

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=cm_colors[0], label='Slavic'),
    Patch(facecolor=cm_colors[1], label='Romance'),
    Patch(facecolor=cm_colors[2], label='Germanic'),
    Patch(facecolor=cm_colors[3], label='Chinese')
]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig("/home/students/innes/ba2/LLM-aspect/img/lang_level.jpeg", dpi=300)
plt.show()
