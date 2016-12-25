from conllu.parser import parse
import nltk


class NgramTagger:

    # INITIAL SET-UP

    def __init__(self, trainingFile, n=3):
        # Set up
        self.trainingData = self.__process(trainingFile)
        self.n = n
        self.ngramLists = []  # Will contain ngrams for every n from 1 to self.n
        self.ngramDicts = []  # Will contain dictionaries for every n from 1 to self.n

        # Train tagger
        for i in range(1, n + 1):  # Skip 0 and include n
            self.ngramLists.append(self.__ngrams(self.trainingData, i))
            self.ngramDicts.append(self.__ngramDict(self.ngramLists[i - 1], i))


    # UTILITY FUNCTIONS

    # Extracts word form and POS tag from conllu file
    def __process(self, file):
        fullFile = parse(open(file, encoding='UTF8').read())
        return [(word['form'].lower(), word['upostag']) for sentence in fullFile for word in sentence]

    # Creates ngrams for any n > 0
    def __ngrams(self, tokens, n):
        ngs = []
        for i in range(len(tokens) - n + 1):
            ngs.append(tuple(tokens[i:i + n]))
        # print(ngs)  # debug
        return ngs

    # Creates dict with word and n-1 previous tags as key and a frequency distribution of words that follow as the value
    def __ngramDict(self, ngrams, n):
        ret = {}
        for ngram in ngrams:
            # Generate key and tag
            key = self.__generateKey(ngram)
            tag = ngram[n - 1][1]

            # Initialize frequency dictionary for key
            ret[key] = ret.get(key, {})
            # Add new information to dictionary
            ret[key][tag] = ret[key].get(tag, 0) + 1
        return ret

    def __generateKey(self, ngram):
        key = []
        for i in range(len(ngram) - 1):
            key.append(ngram[i][1])  # Add n-1 previous tags to key
        key.append(ngram[-1][0])  # Add word to key
        return tuple(key)  # Make into a tuple to use

    # ACCESSORS

    # Get data used to train model
    def getTrainingData(self):
        return self.trainingData

    # TAGGING METHODS

    # Returns percent of words tagged correctly
    # Assumes test file is in conllu format
    # Use verbose argument to print out all incorrectly tagged words and the most common errors made by tagger.
    # Use errorsToDisplay argument to change how many of the most common errors are printed out
    def test(self, testFile, verbose=False, errorsToDisplay=10):
        # Get list of words and have tagger tag them
        answers = self.__process(testFile)
        testWords = [word for (word, tag) in answers]
        tagged = self.tag(testWords)
        # print(tagged)  # debug

        # Count correctly tagged words.
        # If verbose is on, count occurrences of specific errors and print out incorrectly tagged words.
        correct = 0
        if verbose:
            print("Incorrect Tags:")
            problems = {}
        for i in range(len(testWords)):
            if answers[i][1] == tagged[i][1]:
                correct += 1
            elif verbose:
                print(answers[i][0], ":", answers[i][1], "tagged as", tagged[i][1])
                problems[(answers[i][1], tagged[i][1])] = problems.get((answers[i][1], tagged[i][1]), 0) + 1
        # If verbose is on, print out first ten (or errorsToDisplay) most common errors and what percent of incorrect
        # tags they're each responsible for.
        if verbose:
            sortedErrors = [tup for tup in sorted(problems.keys(), key=lambda x: problems[x], reverse=True)]
            print("\nCauses of errors:")
            display = min(errorsToDisplay, len(sortedErrors))  # Prevents out-of-bounds exceptions
            for err in sortedErrors[:display]:
                print("" + err[0] + " tagged as " + err[1] + ": %f" % (problems[err] / (len(testWords) - correct)))
        # Return percent of words labelled correctly
        return correct / len(testWords)

    # Tags sentences in a file
    # Assumes file is in a regular text format
    def tagFile(self, file):
        words = nltk.tokenize.wordpunct_tokenize(file)
        return self.tag(words)

    # Tags a list of words
    def tag(self, words):
        ret = []
        for i in range(len(words)):
            if i < self.n:
                key = self.__generateKey(ret + [(words[i], '')])
                ret.append(self.tagWord(key))
            else:
                key = self.__generateKey(ret[i - self.n + 1:] + [(words[i], '')])
                ret.append(self.tagWord(key))
            # print(ret)  # debug
        return ret

    # Recursive function to tag words.
    # Key must be a tuple of length n containing previous n-1 tags and word to tag
    #   Exception is if n = 1, then the key is just the word to tag
    def tagWord(self, key):
        n = len(key)
        freqDist = self.ngramDicts[n-1].get(key)  # Access the correct dictionary and get value of key
        if freqDist:  # If we found an answer
            # print("Found entry:", key)
            return key[-1], max(freqDist.keys(), key=lambda x: freqDist[x])  # Get the most likely tag for this context
        elif n > 1:  # If not and we aren't already at unigram tagger
            # print("Backed off:", key)
            return self.tagWord(key[1:])  # Back off
        else:  # Default tagger
            # print("Went to default tag:", key)  # debug
            return key[0], "NOUN"


# Runs example using universal dependencies train and test set for French and a plaintext test file
if __name__ == "__main__":
    tagger = NgramTagger("fr-ud-train.conllu", 3)
    print(tagger.test("fr-ud-test.conllu", verbose=True))
    # print(tagger.tagFile('test.txt'))
