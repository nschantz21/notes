"""
Analyzing Sentence Structure
"""

"""
Acceptor Using Well-Formed Substring Table
"""
import nltk

def init_wfst(tokens, grammar):
    """
    create an (n-1) x (n-1) matrix as a list of lists in Python, and initialize it with the lexical categories of each token
    """
    numtokens = len(tokens)
    wfst = [[None for i in range(numtokens+1)] for j in range(numtokens+1)]
    
    for i in range(numtokens):
        productions = grammar.productions(rhs=tokens[i])
        print productions
        wfst[i][i+1] = productions[0].lhs()
    return wfst
    
def complete_wfst(wfst, tokens, grammar, trace=False):
    index = dict((p.rhs(), p.lhs()) for p in grammar.productions())
    numtokens = len(tokens)
    for span in range(2, numtokens+1):
        for start in range(numtokens+1-span):
            end = start + span
            for mid in range(start+1, end):
                nt1, nt2 = wfst[start][mid], wfst[mid][end]
                if nt1 and nt2 and (nt1,nt2) in index:
                    wfst[start][end] = index[(nt1,nt2)]
                    if trace:
                        print "[%s] %3s [%s] %3s [%s] ==> [%s] %3s [%s]" % \
                        (start, nt1, mid, nt2, end, start, index[(nt1,nt2)], end)
    return wfst

def display(wfst, tokens):
    """
    Pretty Prints the WFST
    """
    print '\nWFST ' + ' '.join([("%-4d" % i) for i in range(1, len(wfst))])
    for i in range(len(wfst)-1):
        print "%d   " % i,
        for j in range(1, len(wfst)):
            print "%-4s" % (wfst[i][j] or '.'),
        print



# not working
def head_complement_filter(head, complement, tree):
    child_nodes = [child.node for child in tree
                   if isinstance(child, nltk.Tree)]
    return  (tree.node == head) and (complement in child_nodes)
    
def filter(tree):
    child_nodes = [child.node for child in tree if isinstance(child, nltk.Tree)]
    return  (tree.node.label() == 'VP') and ('S' in child_nodes)

if __name__ == '__main__':
    
    # context free grammar parser
    groucho_grammar = nltk.CFG.fromstring(
        """
        S -> NP VP
        PP -> P NP
        NP -> Det N | Det N PP | 'I'
        VP -> V NP | VP PP
        Det -> 'an' | 'my'
        N -> 'elephant' | 'pajamas'
        V -> 'shot'
        P -> 'in'
        """
    )
    
    # Sentence to tokenize
    tokens = "I shot an elephant in my pajamas".split()
    
    wfst0 = init_wfst(tokens, groucho_grammar)
    display(wfst0, tokens)
    
    wfst1 = complete_wfst(wfst0, tokens, groucho_grammar)
    display(wfst1, tokens)
    
    wfst1 = complete_wfst(wfst0, tokens, groucho_grammar, trace=True)