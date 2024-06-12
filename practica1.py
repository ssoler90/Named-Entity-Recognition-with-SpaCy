# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 12:07:36 2023

@author: soler
"""

import spacy # importamos spacy

# creamos un objeto con el modelo preentrenado "es_core_news_sm"
nlp = spacy.load("es_core_news_lg") 

# Cargamos los datos de prueba
with open ("data/esp.testb.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()
    
# Separamos de las palabras de las palabras de sus etiquetas
tokens = []

for linea in lines:
    partes = linea.strip().split()
    if len(partes) == 2:
        palabra, etiqueta = partes
        tokens.append((palabra, etiqueta))
        
# creamos un doc con spacy a partir de las palabras ya separadas
text = spacy.tokens.Doc(nlp.vocab, words=[token[0] for token in tokens])

# comprobamos que text está en texto plano
print(text) 

# pasamos la funcion nlp al texto base
doc = nlp(text)

# comprobamos que doc.text para ver que a simple vista es igual que text
print(doc.text)

#Comprobamos que tengan la misma longitud
print(len(doc) == len(tokens))

# creamos la función siguiente para estar seguros que ambos textos coinciden
def are_identical(doc, tokens):
    """
    Devuelve True si ambos textos son idénticos
    """
    i = 0
    add = 0
    for word in doc:
        if word.text == tokens[i][0]:
            add += 1
    
        i += 1

   
    if add == len(doc):
        return True
    else:
        return False
    
are_identical(doc, tokens)

# Podemos visualizar como quedaría el etiquetado hecho por el modelo
for token in doc:
    print(token.text ,token.ent_iob_+"-"+token.ent_type_)


# importamos la funcion conlleval
import conlleval

# Creamos una lista con los valores verdaderos etiquetados
true_seqs =[]
for token in tokens:
    true_seqs.append(token[1])

# Creamos una lista con los valores predichos  
pred_seqs = []
for token in doc:
    pred_seqs.append(token.ent_iob_+"-"+token.ent_type_)
    
# evaluamos el modelo
conlleval.evaluate(true_seqs, pred_seqs)

# función para ver los falsos positivos
def false_positives(entity):
    i=0
    for token in doc:
        if token.ent_type_ == entity and not(tokens[i][1] == 'B-'+ entity or tokens[i][1] == 'I-'+ entity):
            print(token.text)
        i +=1

# funcion para ver los falsos negativos        
def false_negatives(entity):
    i=0
    for token in doc:
        if token.ent_type_ != entity and (tokens[i][1] == 'B-' + entity or tokens[i][1] == 'I-' + entity):
            print(token.text, ' ',  token.ent_iob_+"-"+token.ent_type_)
        i +=1

# echamos un vistazo a a los falsos positivos y negativos
false_positives('LOC')
false_negatives('LOC')


'''
-----------------PARTE-VOLUNTARIA----------------------------------------------
'''

# Cargamos los datos de prueba
with open ("iron_maiden.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()

# Separamos de las palabras de las palabras de sus etiquetas
tokens = []

for linea in lines:
    partes = linea.strip().split()
    if len(partes) == 2:
        palabra, etiqueta = partes
        tokens.append((palabra, etiqueta))
        
# creamos un doc con spacy a partir de las palabras ya separadas
text = spacy.tokens.Doc(nlp.vocab, words=[token[0] for token in tokens])

# pasamos la funcion nlp al texto base
doc = nlp(text)

# comprobamos que sean identicos
are_identical(doc, tokens)

true_seqs =[]
for token in tokens:
    true_seqs.append(token[1])

pred_seqs = []
for token in doc:
    pred_seqs.append(token.ent_iob_+"-"+token.ent_type_)
    
conlleval.evaluate(true_seqs, pred_seqs)

