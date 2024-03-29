from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import sys
import unicodedata
import codecs
from io import open
import itertools
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

corpus_name = "comments"
corpus = os.path.join("../../data", corpus_name)
save_dir = os.path.join("../../data", "save")

MAX_LENGTH = 10 # Maximum sentence length to consider

# Define path to new file
datafile = os.path.join(corpus, "formatted_data.txt")

# Only run all this processing if starting new model
if len(sys.argv) < 2:

    # Show the data real quick
    def printLines(file, n=10):
        with open(file, 'rb') as datafile:
            lines = datafile.readlines()
        for line in lines[:n]:
            print(line)

    printLines(os.path.join(corpus, "comments_long_cleaned.txt"))

    # Return array of comment threads -  the 0 index of each is parent comment
    def splitByCommentThread(fileName):
        threads = []
        with open(fileName, 'r') as f:
            current_thread = []
            for line in f:
                if line.find("$BREAK") == -1:
                    current_thread.append(line)
                else: 
                    threads.append(current_thread)
                    current_thread = []
        return threads
        
    # Return array of [x, y] arrays containing comment[0] and response[1] 
    def processThreads(threads):
        conversations = [] 
        for thread in threads:
            # first see if comment has more than one 
            if len(thread) > 1:
                # First comment in thread is root comment. current_conversation also contains root_comment
                root_comment = thread[0]
                for comment in thread:
                    if comment != root_comment:
                        current_conversation = [root_comment, comment] 
                        conversations.append(current_conversation)
        return conversations



    # WRITE 

    # Process, baby
    threads = splitByCommentThread(os.path.join(corpus, "comments_long_cleaned.txt"))
    conversations = processThreads(threads)

    # Get rid of newlines
    for i in range(len(conversations)):
        conversations[i][0] = conversations[i][0].replace("\n", "")
        conversations[i][1] = conversations[i][1].replace("\n", "")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Write to .txt
    # TODO: Get this to write to a single line. Not sure why it won't work
    with open(datafile, 'a', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in conversations:
            writer.writerow(pair)

    # Sample of kinda formatted data 
    printLines(datafile)



# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start of sentence token
EOS_token = 2  # End of sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k) 

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        # Add words back that weren't trimmed
        for word in keep_words:
            self.addWord(word)

# Preprocessing bs
        
# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s    

# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
    print("Reading lines...")
    # Read the file and split into lines
    lines = []
    with open(datafile) as file:
        for line in file: 
            line = line.strip()
            lines.append(line)
    # lines = open(datafile, encoding='utf-8').\
    #     read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs  

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filterPair(p):
    # Shitty hack to handle pairs that aren't correct (should be handled in data)
    if len(p) == 2:
        # Input sequences need to preserve the last word for EOS token
        return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH 

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)] 

# Using the above functions, returned a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data ...")
    voc, pairs = readVocs(datafile, corpus_name) 
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words ...")
    for pair in pairs: 
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

# Load/Assemble voc and paris
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

# Optionally get rid of rarely used words also
MIN_COUNT = 3   # Minimum word count threshold for trimming

def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

# trim voc and pairs
pairs = trimRareWords(voc, pairs, MIN_COUNT)    


# Now we gotta convert sentences to tensors (with 0 padding) and do math on them jawns

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

# Used in output var. Replaces all zero padded values with 0 and all other values with 1
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq: 
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m    

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask (binary jawn), and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len 


# # Example for validation
# small_batch_size = 5
# batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
# input_variable, lengths, target_variable, mask, max_target_len = batches

# print("input_variable:", input_variable)
# print("lengths:", lengths)
# print("target_variable:", target_variable)
# print("mask:", mask)
# print("max_target_len:", max_target_len)