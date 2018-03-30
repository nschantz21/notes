"""
Extracting Information from Text
"""
import nltk, re, pprint

def ie_preprocess(document):
    """
    connects together NLTK's default sentence segmenter, word tokenizer, and part-of-speech tagger
    """
    sentences = nltk.sent_tokenize(document)
    sentences_list = [nltk.word_tokenize(sent) for sent in sentences]
    sentences_positions = [nltk.pos_tag(sent) for sent in sentences_list]
    return sentences_positions
    
def simple_regex_chunker(sentence, regex):
    """
    Simple Regular Expression Based NP Chunker.
    """
    cp = nltk.RegexpParser(regex)
    result = cp.parse(sentence)
    print result
    result.draw()



class Unigram_Bigram_Chunker(nltk.ChunkParserI):
    """
    Noun Phrase Chunking with a Unigram Tagger
    """
    def __init__(self, train_sents, chunker_type = 'uni'):
        # trains the chunker
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        if(chunker_type == 'uni'):
            self.tagger = nltk.UnigramTagger(train_data)
        if(chunker_type == 'bi'):
            self.tagger = nltk.BigramTagger(train_data)
        

    def parse(self, sentence):
        # Unigram chunker
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        # return tree representation of chunker
        return nltk.chunk.conlltags2tree(conlltags)





class ConsecutiveNPChunkTagger(nltk.TaggerI):
    """
    Noun Phrase Chunking with a Consecutive Classifier
    """
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) # [_consec-use-fe]
                train_set.append( (featureset, tag) )
                history.append(tag)
        # max entropy classifier
        self.classifier = nltk.MaxentClassifier.train(
            train_set, algorithm='megam', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        # turns tagged sentences into ((word, tag), chunk)
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        # maps the chunk tree into tag sequences
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        # tags sentences
        tagged_sents = self.tagger.tag(sentence)
        # reformat tags
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        # converts tag sequence provided by the tagger back into a chunk tree.
        return nltk.chunk.conlltags2tree(conlltags)

def npchunk_features(sentence, i , history):
    
    """
    Feature Extractor - called from ConsecutiveNPChunkTagger.tag()
    Provides part-of-speech tags
    history parameter is unused here
    """
    
    def tags_since_dt(sentence, i):
        """
        string describing the set of all pos tags that have been encountered since the most recent determiner
        """
        tags= set()
        for word, pos, in sentence[:i]:
            if pos == 'DT':
                tags = set()
            else:
                tags.add(pos)
        return '+'.join(sorted(tags))
    
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence) - 1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {
        "pos": pos,
        "word": word,
        "prevpos": prevpos,
        "nextpos": nextpos,
        "prevpos+pos": "%s+%s" % (prevpos, pos), # paired features
        "pos+nextpos": "%s+%s" % (pos, nextpos), # complex contextual feautures
        "tags-since-dt": tags_since_dt(sentence, i) # string describing the set of all pos tags that have been encountered since the most recent determiner
    }




def traverse(t):
    """
    Recursive Function to traverse a tree
    
    """
    try:
        t.label()
    except AttributeError:
        print t,
    else:
        # Now we know that t.node is defined
        print '(', t.label(),
        for child in t:
            traverse(child)
        print ')',






if __name__ == '__main__':
    
    # simple_regex_chunker
    sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"), ("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    simple_regex_chunker(sentence, grammar)
    