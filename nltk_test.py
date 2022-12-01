import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.book import *
from nltk import FreqDist

# author: adrian muth
# based off of https://realpython.com/nltk-nlp-python/

# tokenizing

example_string = """
Muad'Dib learned rapidly because his first training was in how to learn.
And the first lesson of all was the basic trust that he could learn.
It's shocking to find how many people do not believe they can learn,
and how many more believe learning to be difficult."""

print(sent_tokenize(example_string))
print("BREAK")
print(word_tokenize(example_string))

# stop words

worf_quote = "Sir, I protest.  I am not a merry man!"
words_in_quote = word_tokenize(worf_quote)
print(words_in_quote)

stop_words = set(stopwords.words("english"))
filtered_list = []
for word in words_in_quote:
    if word.casefold() not in stop_words:
        filtered_list.append(word)
        
print(filtered_list)

# stemming

stemmer = PorterStemmer()
string_for_stemming = """
The crew of the USS Discovery discovered many discoveries.
Discovering is what explorers do."""

words = word_tokenize(string_for_stemming)
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)

# part of speech tagging

sagan_quote = """
If you wish to make an apple pie from scratch,
you must first invent the universe."""
words_in_sagan_quote = word_tokenize(sagan_quote)

print(nltk.pos_tag(words_in_sagan_quote))

# lemmatizing
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("scarves"))

string_for_lemmatizing = "The friends of DeSoto love scarves."
words = word_tokenize(string_for_lemmatizing)
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print(lemmatized_words)

print(lemmatizer.lemmatize("worst", pos="a"))

# chunking

lotr_quote = "It's a dangerous business, Frodo, going out your door."
words_in_lotr_quote = word_tokenize(lotr_quote)
lotr_pos_tags = nltk.pos_tag(words_in_lotr_quote)

grammar = "NP: {<DT>?<JJ>*<NN>}"
chunk_parser = nltk.RegexpParser(grammar)

tree = chunk_parser.parse(lotr_pos_tags)
tree.draw()

grammar2 = """
Chunk: {<.*>+}
    }<JJ>{"""
chunk_parser2 = nltk.RegexpParser(grammar2)

tree2 = chunk_parser2.parse(lotr_pos_tags)
tree2.draw()

# named entity recognition (NER)

tree3 = nltk.ne_chunk(lotr_pos_tags)
tree3.draw()

quote = """
Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
for countless centuries Mars has been the star of war—but failed to
interpret the fluctuating appearances of the markings they mapped so well.
All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
and then by other observers. English readers heard of it first in the
issue of Nature dated August 2."""

def extract_ne(quote):
    words = word_tokenize(quote)
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
        )

print(extract_ne(quote))

# analyzing text

# concordance
print(text8.concordance("man"))
print(text8.concordance("woman"))

# dispersion plot
text8.dispersion_plot(["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"])
text2.dispersion_plot(["Allenham", "Whitwell", "Cleveland", "Combe"])

# frequency distribution
frequency_distribution = FreqDist(text8)
print(frequency_distribution)
print(frequency_distribution.most_common(20))

meaningful_words = [word for word in text8 if word.casefold() not in stop_words]
frequency_distribution = FreqDist(meaningful_words)
print(frequency_distribution.most_common(20))

frequency_distribution.plot(20, cumulative=True)

# finding collocations
print(text8.collocations())

lemmatized_words = [lemmatizer.lemmatize(word) for word in text8]
new_text = nltk.Text(lemmatized_words)
print(new_text.collocations())