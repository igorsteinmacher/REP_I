import spacy
from spacy.lang.en import English

nlp = English()
ruler = nlp.add_pipe("entity_ruler").from_disk("./patterns.jsonl")

doc = nlp("You can git pull to pull-request your pull request.")
print([(ent.text, ent.label_, ent.ent_id_) for ent in doc.ents])
