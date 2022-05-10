import decimal
from decimal import Decimal
import random
import sys
#import pyae


class ArithmeticEncoding:
    """
    ArithmeticEncoding is a class for building arithmetic encoding.
    """

    #   Receives alphabet as char list
    def __init__(self, alphabet):
        #self.probability_table = self.get_probability_table(frequency_table)
        self.alphabet = alphabet

    def get_initial_probability_table(self):
        """
        Table gives initial probabilities for first char. Probabilities were given by professor
        """
        return {'a': 0.5, 'b': 0.3, 'c': 0.1, 'd':0.1}

    def get_probability_table_original(self, frequency_table):
        """
        Calculates the probability table out of the frequency table.
        """
        total_frequency = sum(list(frequency_table.values()))

        probability_table = {}
        for key, value in frequency_table.items():
            probability_table[key] = value / total_frequency

        return probability_table

    def get_probability_table(self, previous_sequence):
        """
        Calculates the probability table out of the self.alphabet and previously encoded sequence
        """
        #total_frequency = sum(list(frequency_table.values()))

        probability_table = {}
        for char in alphabet:
            #returns probability for current char
            probability_table[char] = self.laplace(char, previous_sequence, self.alphabet )

        return probability_table

    # Returns conditional probability of character occuring after previous(already encoded) sequence
    #   current_char: char, prev_seq: string, alphabet: list of char
    def laplace(self, current_char, previous_sequence, alphabet):
        # # of current characters in previous seq
        fb = previous_sequence.count(current_char)

        # create count of occrances in previos seq for whole alphabet
        fa_to_fn = []
        for letter in alphabet:
            fa_to_fn.append(previous_sequence.count(letter))

        return (fb + 1) / (sum(fa_to_fn) + len(alphabet))

    def get_encoded_value(self, encoder):
        """
        After encoding the entire message, this method returns the single value that represents the entire message.
        """
        last_stage = list(encoder[-1].values())
        last_stage_values = []
        for sublist in last_stage:
            for element in sublist:
                last_stage_values.append(element)

        last_stage_min = min(last_stage_values)
        last_stage_max = max(last_stage_values)

        return (last_stage_min + last_stage_max) / 2

    def process_stage(self, probability_table, stage_min, stage_max):
        """
        Processing a stage in the encoding/decoding process. Returns probability distribution of current interval
        """
        stage_probs = {}
        stage_domain = stage_max - stage_min
        for term_idx in range(len(probability_table.items())):
            term = list(probability_table.keys())[term_idx]
            term_prob = Decimal(probability_table[term])
            cum_prob = term_prob * stage_domain + stage_min
            stage_probs[term] = [stage_min, cum_prob]
            stage_min = cum_prob
        return stage_probs

    def encode(self, msg):
        """
        Encodes a message.
        """

        encoder = []

        stage_min = Decimal(0.0)
        stage_max = Decimal(1.0)

        for msg_term_idx in range(len(msg)):
            probability_table = self.get_probability_table(msg[0:msg_term_idx])
            if msg_term_idx == 0:
                probability_table = self.get_initial_probability_table()
            stage_probs = self.process_stage(probability_table, stage_min, stage_max)

            msg_term = msg[msg_term_idx]
            stage_min = stage_probs[msg_term][0]
            stage_max = stage_probs[msg_term][1]

            encoder.append(stage_probs)

        stage_probs = self.process_stage(probability_table, stage_min, stage_max)
        encoder.append(stage_probs)

        encoded_msg = self.get_encoded_value(encoder)

        return encoder, encoded_msg

    def decode(self, encoded_msg, msg_length):
        """
        Decodes a message. encoded_msg: real number, len of encoded string
        Kako sploh odkodirati adaptive msg?
        """

        decoder = []
        decoded_msg = ""

        stage_min = Decimal(0.0)
        stage_max = Decimal(1.0)

        for idx in range(msg_length):
            probability_table = self.get_probability_table(decoded_msg)
            if decoded_msg == "":
                probability_table = self.get_initial_probability_table()
            stage_probs = self.process_stage(probability_table, stage_min, stage_max)

            #msg_term: char in alphabet, value: chars l and probability for current interval
            for msg_term, value in stage_probs.items():
                #break if we are in the right interval
                if encoded_msg >= value[0] and encoded_msg <= value[1]:
                    break

            decoded_msg = decoded_msg + msg_term
            stage_min = stage_probs[msg_term][0]
            stage_max = stage_probs[msg_term][1]

            decoder.append(stage_probs)

        stage_probs = self.process_stage(probability_table, stage_min, stage_max)
        decoder.append(stage_probs)

        return decoder, decoded_msg



# --------- HUFFMAN
# A Huffman Tree Node
class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        # probability of symbol
        self.prob = prob

        # symbol
        self.symbol = symbol

        # left node
        self.left = left

        # right node
        self.right = right

        # tree direction (0/1)
        self.code = ''


""" A helper function to print the codes of symbols by traveling Huffman Tree"""
codes = dict()


def Calculate_Codes(node, val=''):
    # huffman code for current node
    newVal = val + str(node.code)

    if (node.left):
        Calculate_Codes(node.left, newVal)
    if (node.right):
        Calculate_Codes(node.right, newVal)

    if (not node.left and not node.right):
        codes[node.symbol] = newVal

    return codes


""" A helper function to calculate the probabilities of symbols in given data"""


def Calculate_Probability(data):
    symbols = dict()
    for element in data:
        if symbols.get(element) == None:
            symbols[element] = 1
        else:
            symbols[element] += 1
    return symbols


""" A helper function to obtain the encoded output"""


def Output_Encoded(data, coding):
    encoding_output = []
    for c in data:
        #  print(coding[c], end = '')
        encoding_output.append(coding[c])

    string = ''.join([str(item) for item in encoding_output])
    return string


""" A helper function to calculate the space difference between compressed and non compressed data"""


def Total_Gain(data, coding):
    before_compression = len(data) * 8  # total bit space to stor the data before compression
    after_compression = 0
    symbols = coding.keys()
    for symbol in symbols:
        count = data.count(symbol)
        after_compression += count * len(coding[symbol])  # calculate how many bit is required for that symbol in total
    print("Space usage before compression for Huffman coding (in bits):", before_compression)
    print("Space usage after compression for Huffman coding  (in bits):", after_compression)

def compression_ratio_huffman(data, coding):
    before_compression = len(data) * 8  # total bit space to stor the data before compression
    after_compression = 0
    symbols = coding.keys()
    for symbol in symbols:
        count = data.count(symbol)
        after_compression += count * len(coding[symbol])  # calculate how many bit is required for that symbol in total
    print("Compression ratio for huffman coding: {ratio}".format(
        ratio=after_compression / before_compression))


def Huffman_Encoding(data):
    symbol_with_probs = Calculate_Probability(data)
    symbols = symbol_with_probs.keys()
    probabilities = symbol_with_probs.values()
    print("symbols: ", symbols)
    print("probabilities: ", probabilities)

    nodes = []

    # converting symbols and probabilities into huffman tree nodes
    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))

    while len(nodes) > 1:
        # sort all the nodes in ascending order based on their probability
        nodes = sorted(nodes, key=lambda x: x.prob)
        # for node in nodes:
        #      print(node.symbol, node.prob)

        # pick 2 smallest nodes
        right = nodes[0]
        left = nodes[1]

        left.code = 0
        right.code = 1

        # combine the 2 smallest nodes to create new node
        newNode = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = Calculate_Codes(nodes[0])
    print("symbols with codes", huffman_encoding)
    Total_Gain(data, huffman_encoding)
    compression_ratio_huffman(data, huffman_encoding)
    encoded_output = Output_Encoded(data, huffman_encoding)
    return encoded_output, nodes[0]


def Huffman_Decoding(encoded_data, huffman_tree):
    tree_head = huffman_tree
    decoded_output = []
    for x in encoded_data:
        if x == '1':
            huffman_tree = huffman_tree.right
        elif x == '0':
            huffman_tree = huffman_tree.left
        try:
            if huffman_tree.left.symbol == None and huffman_tree.right.symbol == None:
                pass
        except AttributeError:
            decoded_output.append(huffman_tree.symbol)
            huffman_tree = tree_head

    string = ''.join([str(item) for item in decoded_output])
    return string

# Creates a sequence(as string) of length n with characters from alphabet
#   alphabet: list of charachters or string of characters
#   n: len of sequence
def create_sequence(alphabet, n):
    """
        A helper function to obtain the encoded output, Creates a sequence(as string) of length n with characters from alphabet
        alphabet: list of charachters or string of characters
         n: len of sequence
    """
    if (type(alphabet) == list):
        # list to string
        alphabet_as_string = ''.join(alphabet)

    # random sample of len n
    return ''.join(random.choice(alphabet_as_string) for i in range(n))

def total_gain_aac(sequence, encoded_number):
    """
        Comparison of required bits for initial string versus compressed real number
        Returns tuple of before and after compression
    """
    before_compression = len(sequence) * 8  # total bit space to stor the data before compression
    after_compression = sys.getsizeof(encoded_number)
    # print("Space usage before compression (in bits):", before_compression)
    # print("Space usage after compression (in bits):", after_compression)
    return (before_compression, after_compression)

if __name__ == '__main__':
    decimal.getcontext().prec = 70

    print("---------Adaptive Arithmetic encoding and decoding start----------------")
    alphabet = ['a', 'b', 'c', 'd']
    AE = ArithmeticEncoding(alphabet)

    original_msg_1 = "aababdc"
    original_msg = create_sequence(alphabet, 100)
    print("Original Message: {msg}".format(msg=original_msg))

    encoder, encoded_msg = AE.encode(msg=original_msg)
    print("Encoded Message: {msg}".format(msg=encoded_msg))

    decoder, decoded_msg = AE.decode(encoded_msg=encoded_msg,
                                     msg_length=len(original_msg))
    print("Decoded Message: {msg}".format(msg=decoded_msg))

    print("Message Decoded Successfully? {result}".format(result=original_msg == decoded_msg))

    before_compression = total_gain_aac(original_msg, encoded_msg)[0]
    after_compression = total_gain_aac(original_msg, encoded_msg)[1]
    print("Space usage before compression for adaptive arithmetic coding (in bits):", before_compression)
    print("Space usage after compression for adaptive arithmetic coding(in bits):", after_compression)
    #get compression ratio
    print("Compression ratio for adaptive arithmetic coding: {ratio}".format(ratio=after_compression/before_compression) )

    print("---------Adaptive Arithmetic encoding and decoding end----------------")

    ### -----huffman ----
    print()
    print("---------Huffman encoding and decoding start----------------")
    encoding, tree = Huffman_Encoding(original_msg)
    print("Encoded output", encoding)
    print("Decoded Output", Huffman_Decoding(encoding, tree))
    print("---------Huffman encoding and decoding end----------------")

    #DODAJ DOLZINO ARITMETICNEGA
    #DODAJ LAPLACE