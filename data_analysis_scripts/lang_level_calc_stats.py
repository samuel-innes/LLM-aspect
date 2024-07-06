"""
Calculate statistics on language-level entropy data
"""

import numpy as np
from scipy import stats
def read_file(file_path):
    tuples_list = []
    entropies = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):
            entropy_line = lines[i].strip()
            verb_line = lines[i + 1].strip()
            class_line = lines[i + 2].strip()
            
            # Extract entropy value
            entropy = round(float(entropy_line.split(': ')[1]), 5)
            
            # Extract verb
            verb = verb_line
            
            # Extract class
            predicted_class = class_line
            
            # Create a tuple and add it to the list
            tuples_list.append((verb, entropy, predicted_class))
            entropies.append(entropy)
    
    return tuples_list, entropies

if __name__ == '__main__':
    # TED2013
    _, de = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_de.txt")
    _, en = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_en.txt")
    _, es = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_es.txt")
    _, fr = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_fr.txt")
    _, it = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_it.txt")
    _, nl = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_nl.txt")
    _, pl = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_pl.txt")
    _, pt = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_pt.txt")
    _, ro = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_ro.txt")
    _, ru = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_ru.txt")
    _, sl = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_sl.txt")
    _, zh = read_file("/home/students/innes/ba2/output_files/BERT/TED2013/verb_entropies_zh.txt")
    print("TED2013")
    print("Mean of Chinese entropy: ", np.mean(zh), "variance: ", np.std(zh))
    print("\n")

    print("Mean of English entropy: ", np.mean(en), "variance: ", np.var(en))
    print("Mean of German entropy: ", np.mean(de), "variance: ", np.std(de))
    print("Mean of Dutch entropy: ", np.mean(nl), "variance: ", np.std(nl))
    print("\n")

    print("Mean of Italian entropy: ", np.mean(it), "variance: ", np.std(it))
    print("Mean of Romanian entropy: ", np.mean(ro), "variance: ", np.std(ro))
    print("Mean of Spanish entropy: ", np.mean(es), "variance: ", np.std(es))
    print("Mean of Portugese entropy: ", np.mean(pt), "variance: ", np.std(pt))
    print("\n")

    print("Mean of Polish entropy: ", np.mean(pl), "variance: ", np.std(pl))
    print("Mean of Slovene entropy: ", np.mean(sl), "variance: ", np.std(sl))
    print("Mean of Russian entropy: ", np.mean(ru), "variance: ", np.std(ru))

    # TED2020
    print("\nTED2020")

    _, ar = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_ar.txt")
    _, be = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_be.txt")
    _, bg = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_bg.txt")
    _, ca = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_ca.txt")
    _, cs = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_cs.txt")
    _, da = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_da.txt")
    _, de = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_de.txt")
    _, el = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_el.txt")
    _, es = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_es.txt")
    _, en = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_en.txt")    
    _, et = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_et.txt")    
    _, fi = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_fi.txt")
    _, fr = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_fr.txt")
    _, hr = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_hr.txt")
    _, hu = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_hu.txt")
    _, it = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_it.txt")
    _, isl = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_is.txt")
    _, lt = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_lt.txt")
    _, lv = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_lv.txt")
    _, nb = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_nb.txt")
    _, nn = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_nn.txt")
    _, nl = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_nl.txt")
    _, pl = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_pl.txt")
    _, pt = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_pt.txt")
    _, ru = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_ru.txt")
    _, ro = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_ro.txt")
    _, sk = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_sk.txt")
    _, sl = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_sl.txt")
    _, sv = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_sv.txt")
    _, uk = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_uk.txt")
    _, vi = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_vi.txt")
    _, zh = read_file("/home/students/innes/ba2/output_files/BERT/TED2020/verb_entropies_zh.txt")

    print("Mean of Arabic entropy: ", np.mean(ar), "variance: ", np.std(ar))
    print("Mean of Vietnamese entropy: ", np.mean(vi), "variance: ", np.std(vi))

    print("Mean of Chinese entropy: ", np.mean(zh), "variance: ", np.std(zh))
    print("\n")

    print("Mean of Danish entropy: ", np.mean(da), "variance: ", np.std(da))
    print("Mean of Dutch entropy: ", np.mean(nl), "variance: ", np.std(nl))
    print("Mean of English entropy: ", np.mean(en), "variance: ", np.var(en))
    print("Mean of German entropy: ", np.mean(de), "variance: ", np.std(de))
    print("Mean of Icelandic entropy: ", np.mean(isl), "variance: ", np.std(isl))
    print("Mean of Norwegian Bokmål entropy: ", np.mean(nb), "variance: ", np.std(nb))
    print("Mean of Norwegian Nynorsk entropy: ", np.mean(nn), "variance: ", np.std(sv))
    print("Mean of Swedish entropy: ", np.mean(sv), "variance: ", np.std(sv))
    print("\n")

    print("Mean of Catalan entropy: ", np.mean(ca), "variance: ", np.std(ca))
    print("Mean of French entropy: ", np.mean(fr), "variance: ", np.std(fr))
    print("Mean of Italian entropy: ", np.mean(it), "variance: ", np.std(it))
    print("Mean of Portugese entropy: ", np.mean(pt), "variance: ", np.std(pt))
    print("Mean of Romanian entropy: ", np.mean(ro), "variance: ", np.std(ro))
    print("Mean of Spanish entropy: ", np.mean(es), "variance: ", np.std(es))
    print("\n")

    print("Mean of Finnish entropy: ", np.mean(fi), "variance: ", np.std(fi))
    print("Mean of Estonian entropy: ", np.mean(et), "variance: ", np.std(et))
    print("Mean of Hungarian entropy: ", np.mean(hu), "variance: ", np.std(hu))
    print("\n")

    print("Mean of Greek entropy: ", np.mean(el), "variance: ", np.std(el))
    print("\n")

    print("Mean of Latvian entropy: ", np.mean(lv), "variance: ", np.std(lv))
    print("Mean of Lithuanian entropy: ", np.mean(lt), "variance: ", np.std(lt))
    print("\n")

    print("Mean of Belarussian entropy: ", np.mean(be), "variance: ", np.std(be))
    print("Mean of Bulgarian entropy: ", np.mean(bg), "variance: ", np.std(bg))
    print("Mean of Czech entropy: ", np.mean(cs), "variance: ", np.std(cs))
    print("Mean of Croatian entropy: ", np.mean(cs), "variance: ", np.std(cs))
    print("Mean of Polish entropy: ", np.mean(pl), "variance: ", np.std(pl))
    print("Mean of Russian entropy: ", np.mean(ru), "variance: ", np.std(ru))
    print("Mean of Slovakian entropy: ", np.mean(sk), "variance: ", np.std(sk))
    print("Mean of Slovene entropy: ", np.mean(sl), "variance: ", np.std(sl))
    print("Mean of Ukrainian entropy: ", np.mean(uk), "variance: ", np.std(uk))

    slav = be + bg + cs + pl + ru + sk + sl + uk

    germanic = da +nl + en + de + isl + nb + nn + sv

    romance = ca + fr + it + pt + ro + es

    uralic = fi + et + hu

    print("slav vs germanic", stats.ttest_ind(slav, germanic))
    print("romance vs germanic", stats.ttest_ind(romance, germanic))
    print("slav vs romance", stats.ttest_ind(slav, romance))

