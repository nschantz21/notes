import math
import nltk

def gender_features2(name):
    """
    A Feature Extractor that Overfits Gender Features. The featuresets returned by this feature extractor contain a large number of specific features, leading to overfitting for the relatively small Names Corpus.
    """
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = name.lower().count(letter)
        features["has(%s)" % letter] = (letter in name.lower())
    return features

class DocumentClassifier:
    
    @staticmethod
    def most_informative(document, n = 5):
        """
        To find out which features the classifier found to be most informative
        """
        featuresets = [(document_features(d), c) for (d,c) in documents]
        f_len = len(featuresets)
        # split the feature set in half
        train_set, test_set = featuresets[flen/2:], featuresets[:f_len/2]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print nltk.classify.accuracy(classifier, test_set)
        classifier.show_most_informative_features(n)


    
    @staticmethod
    def document_features(document): # [_document-classify-extractor]
        """
        A feature extractor for document classification, whose features indicate    whether or not individual words are present in a given document.
        """
        document_words = set(document) # [_document-classify-set]
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features



def entropy(labels):
    """
    Calculating the Entropy of a list of labels
    
    Entropy is defined as the sum of the probability of each label times the log probability of that same label.
    """
    freqdist = nltk.FreqDist(labels)
    probs = [freqdist.freq(l) for l in freqdist]
    return -sum([p * math.log(p,2) for p in probs])
    



def pos_features(sentence, i):
    """
    A part-of-speech classifier whose feature detector examines the context in which a word appears in order to determine which part of speech tag should be assigned. In particular, the identity of the previous word is included as a feature.
    """
    features = {"suffix(1)": sentence[i][-1:],
                "suffix(2)": sentence[i][-2:],
                "suffix(3)": sentence[i][-3:]}
    if i == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[i-1]
    return features



 def pos_features(sentence, i, history):
     features = {"suffix(1)": sentence[i][-1:],
                 "suffix(2)": sentence[i][-2:],
                 "suffix(3)": sentence[i][-3:]}
     if i == 0:
         features["prev-word"] = "<START>"
         features["prev-tag"] = "<START>"
     else:
         features["prev-word"] = sentence[i-1]
         features["prev-tag"] = history[i-1]
     return features


def punct_features(tokens, i):
    return {'next-word-capitalized': tokens[i+1][0].isupper(),
            'prevword': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i-1]) == 1}


def segment_sentences(words):
    """
    Classification Based Sentence Segmenter
    """
    start = 0
    sents = []
    for i, word in enumerate(words):
        if word in '.?!' and classifier.classify(punct_features(words, i)) == True:
            sents.append(words[start:i+1])
            start = i+1
    if start < len(words):
        sents.append(words[start:])
    return sents




def rte_features(rtepair):
    """
    "Recognizing Text Entailment" Feature Extractor. The RTEFeatureExtractor class builds a bag of words for both the text and the hypothesis after throwing away some stopwords, then calculates overlap and difference.
    """
    extractor = nltk.RTEFeatureExtractor(rtepair)
    features = {}
    features['word_overlap'] = len(extractor.overlap('word'))
    features['word_hyp_extra'] = len(extractor.hyp_extra('word'))
    features['ne_overlap'] = len(extractor.overlap('ne'))
    features['ne_hyp_extra'] = len(extractor.hyp_extra('ne'))
    return features


def tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word, tag) in sent]
def apply_tagger(tagger, corpus):
    return [tagger.tag(nltk.tag.untag(sent)) for sent in corpus]

class ConsecutivePosTagger(nltk.TaggerI): # [_consec-pos-tagger]
    """
    Part of Speech Tagging with Consecutive Classifier
    """
    
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = pos_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = pos_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

