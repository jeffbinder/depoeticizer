import kenlm
from math import log, ceil

# Maximum number of errors that a word can have based on its length.
MAX_ERRORS_PER_LETTER = 1.0 / 4.0
MAX_ERRORS = 2

# Probability that a single character will be deleted, etc.
ERROR_PROB = 0.2

# ARPA language model.
lm = None

# Ngram probabilities computed using an ARPA-format language model.
def get_ngram_prob(ngram):
    ngram = [tok.lower() for tok in ngram]
    return lm.score(' '.join(ngram))

# Reading probabilities are computed using a simple edit-distance algorithm such as a
# spell-checker might use.
alphabet = 'abcdefghijklmnopqrstuvwxyz'
def isknown(word):
    return word.lower() in lm
def get_reading_probs(tok):
    readings = {}
    def add_tok_readings(tok, prob, n, known):
        if prob == 0.0:
            return
        if n == 0:
            reading_prob = prob
        else:
            reading_prob = prob * (1.0 - ERROR_PROB)
        if known:
            if tok not in readings:
                readings[tok] = reading_prob
        if n == 0:
            return
        # Adapted from http://norvig.com/spell-correct.html.
        s = [(tok[:i], tok[i:]) for i in range(len(tok) + 1)]
        deletions = [a + b[1:] for a, b in s if b]
        transpositions = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
        substitutions = [a + c + b[1:] for a, b in s for c in alphabet if b]
        insertions = [a + c + b for a, b in s for c in alphabet]
        edits = set(deletions + transpositions + substitutions + insertions)
        for edit in edits:
            # Determining the probabilities this way is cheating, really, since I should
            # be accounting for the fact that the observed token is only one of many
            # possible typos that could be generated from a given word.  But my
            # intentions in this project are not serious enough to warrant that much
            # effort.
            add_tok_readings(edit, prob * ERROR_PROB, n-1, isknown(edit))
    add_tok_readings(tok, 1.0, min(int(ceil(MAX_ERRORS_PER_LETTER * len(tok))), MAX_ERRORS), True)
    reading_list = []
    for new_tok in readings:
        logprob = log(readings[new_tok])
        if tok.isupper() and len(tok) > 1:
            new_tok = new_tok.upper()
        elif tok[0].isupper():
            new_tok = new_tok[0].upper() + new_tok[1:].lower()
        reading_list.append((new_tok, logprob))
    return reading_list

def compute_ideal_token_seq(tokens, n=3, get_ngram_prob=get_ngram_prob,
                            get_reading_probs=get_reading_probs):

    # Computes an "ideal" version of the specified token sequence.  This
    # function assumes that the text was generated using a n-gram-based
    # Markov chain - that is, in such a way that the probability of a given
    # token coming next is determined based on what n-1 tokens came before -
    # with a certain probability of a "typo" each time a new token is
    # written.  Two functions must be specified, one giving the probability
    # that a specified n-token sequence will appear, and the other
    # giving a list of all possible words that might have been written as
    # a given token, with probabilities.  Based on the combination of these
    # two models, the function returns the form that the token sequence
    # is most likely to have taken before the "typos."

    # This function uses a variant of the Viterbi algorithm where the possible
    # state sequences are stored in nested dictionaries rather than a matrix.
    # This is because the total number of possible states is enormous (equal
    # to the number of possible n-grams), but the number of states that can
    # produce a given token is relatively small, which would result in an
    # extremely sparse matrix.

    # The first key of d is the most recent token and the second is the token
    # before that, etc.  The value is a pair containing the probability and the
    # complete token sequence up to that point, including the two tokens that
    # are used as keys.

    d = {}

    # The computation for the initial state works a little differently from the
    # later computations because we don't have values for the previous tokens.
    # Instead, we have to compute the probabilities for all possible readings
    # of the first ngram.

    if len(tokens) < n:
        print 'Text must be at least {0} tokens long!'.format(n)
        exit()

    # We want to record all possible values for tokens 2-n, and only
    # the optimal values for the first token.
    def compute_initial_values(i, toks=[], prob=1.0):
        if i == 0:
            max_prob = float("-inf")
            best_tok0 = None
            r = get_reading_probs(tokens[0])
            for tok0, prob0 in r:
                ngram_prob = get_ngram_prob([tok0] + toks)
                final_prob = prob + ngram_prob
                if final_prob > max_prob:
                    max_prob = final_prob
                    best_tok0 = tok0
            return (max_prob, [best_tok0] + toks)
        else:
            d = {}
            r = get_reading_probs(tokens[i])
            #print tokens[i], ':', len(r)
            for toki, probi in r:
                d[toki] = compute_initial_values(i-1, [toki] + toks, probi + prob)
            return d
    d = compute_initial_values(0)

    # Now proceed through the rest of the tokens.  For each one we rebuild d,
    # keeping all options for the previous n-2 tokens and finding the optimal
    # values for the one before that.
    def iterate_possibilities(i, toks, prob, d, tok):
        if isinstance(d, tuple):
            # This should only happen near the beginning when we haven't built up a
            # history.
            prob0, seq0 = d
            ngram_prob = get_ngram_prob(toks)
            final_prob = ngram_prob + prob + prob0
            return (final_prob, seq0 + [tok])
        if i == 0:
            max_prob = float("-inf")
            best_tok0 = None
            best_seq0 = None
            for tok0 in d:
                ngram_prob = get_ngram_prob([tok0] + toks)
                prob0, seq0 = d[tok0]
                final_prob = ngram_prob + prob + prob0
                if final_prob > max_prob:
                    max_prob = final_prob
                    best_tok0 = tok0
                    best_seq0 = seq0
            return (max_prob, best_seq0 + [tok])
        else:
            dnew = {}
            for toki in d:
                dnew[toki] = iterate_possibilities(i-1, [toki] + toks, prob, d[toki], tok)
            return dnew
    for tok in tokens[1:]:
        dnew = {}
        r = get_reading_probs(tok)
        #print tok, ':', len(r)
        for tokn, probn in r:
            dnew[tokn] = iterate_possibilities(n-2, [tokn], probn, d, tokn)
        d = dnew

    # Find the optimal sequence from all the possibilities that remain.
    def extract_ideal_text(i, d, stats):
        if i == 0:
            prob, seq = d
            if prob > stats['max_prob']:
                stats['max_prob'] = prob
                stats['best_seq'] = seq
        else:
            for toki in d:
                extract_ideal_text(i-1, d[toki], stats)
    stats = {'max_prob': float("-inf"), 'best_seq': None}
    extract_ideal_text(n-1, d, stats)

    return stats['best_seq']

def load_language_model(filename):
    global lm
    lm = kenlm.LanguageModel(filename)

def depoeticize(text, n=3, get_ngram_prob=get_ngram_prob,
                get_reading_probs=get_reading_probs):
                  
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'[\w&]([\w&\']*[\w&])?|\S|\s')
    tokens = tokenizer.tokenize(text)
    
    tokens_alpha = [tok for tok in tokens if tok.isalpha()]
    ideal_tokens = compute_ideal_token_seq(tokens_alpha, n, get_ngram_prob,
                                           get_reading_probs)
    
    ideal_text = []
    for tok in tokens:
        if tok.isalpha():
            ideal_text.append(ideal_tokens.pop(0))
        else:
            ideal_text.append(tok)
    return ''.join(ideal_text)


if 0:
    load_language_model('lm_csr_20k_nvp_3gram.binary')
    print depoeticize('''She walks in beauty, like the night
   Of cloudless climes and starry skies;
And all that's best of dark and bright
   Meet in her aspect and her eyes;
Thus mellowed to that tender light
   Which heaven to gaudy day denies.

One shade the more, one ray the less,
   Had half impaired the nameless grace
Which waves in every raven tress,
   Or softly lightens o'er her face;
Where thoughts serenely sweet express,
   How pure, how dear their dwelling-place.

And on that cheek, and o'er that brow,
   So soft, so calm, yet eloquent,
The smiles that win, the tints that glow,
   But tell of days in goodness spent,
A mind at peace with all below,
   A heart whose love is innocent!
''')
