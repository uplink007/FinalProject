import pickle
import logging
import io
import spacy
logging.basicConfig(filename='data_gather_log.log', level=logging.DEBUG)


nlp = spacy.load('en_core_web_sm')

with open('definitions_from_math_world', 'rb') as fp:
    definitions = pickle.load(fp)

thefile = io.open('definitions_from_math_world_text.txt', 'w', encoding='utf-16')
for idx,item in enumerate(definitions):
    try:
        sentence = ''
        doc = nlp(item)
        for token in doc:
            sentence=sentence+" "+token.text
        thefile.write(sentence.rstrip().replace('\n',' '))
        thefile.write('\n')
    except :
        continue

print("Done")



#prime number(basic definitions)




