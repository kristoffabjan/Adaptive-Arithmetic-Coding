# Imports
import random
import string
import decimal


# Creates a sequence(as string) of length n with characters from alphabet
#   alphabet: list of charachters or string of characters
#   n: len of sequence
def create_sequence(alphabet, n):
    if (type(alphabet) == list):
        # list to string
        alphabet_as_string = ''.join(alphabet)

    # random sample of len n
    return ''.join(random.choice(alphabet_as_string) for i in range(n))


# Returns conditional probability of character occuring after previous(already encoded) sequence
#   current_char: char, prev_seq: string, alphabet: list of char
def laplace(current_char, previous_sequence, alphabet):
    # # of current characters in previous seq
    fb = previous_sequence.count(current_char)

    #create count of occrances in previos seq for whole alphabet
    fa_to_fn = []
    for letter in alphabet:
        fa_to_fn.append(previous_sequence.count(letter))

    return (fb + 1) / (sum(fa_to_fn) + len(alphabet))

#   Gets alphabet and current previous sequence and computes conditional probalility for each
#       element of alphabet, according to previous_sequence
#   Returns dictionary of probabilites for each element of alphabet
def conditional_probabilities(alphabet, previous_sequence):
    tuple_list = []
    for i in range(0, len(alphabet)):
        #compute probability of c for existing previous sequence
        #save it as tuple (char, P) and append to tuple_list
        tuple_list.append( (alphabet[i], laplace(alphabet[i], previous_sequence, alphabet)) )
    return tuple_list

def probabilities_before_character(char, tuple_list):
    #index of tuple, containing char in tuple_list
    idx = [i for i, tupl in enumerate(t) if tupl[0] == 'a'][0]
    #idx = next(i for i, (v, *_) in enumerate(tuple_list) if v == char)

    #sum probabilities on previous characters(before char)
    #   sum up second elements in tuples
    tuples_to_sum = tuple_list[0:idx]
    return sum(n for _, n in tuples_to_sum)



# Encode sequence of characters(string) to real number
# sequence: string, alphabel: list of char
def encode(sequence, alphabet):
    #sum of all l() of all characters
    l = decimal.Decimal(0.0)

    #probabilities of encoded charaters to increase l and to get L
    seq_probability = decimal.Decimal(1.0)

    #current product of probabilities of already encoded characters - len of interval
    current_total_probability = decimal.Decimal(1.0)

    for i in range(0,len(sequence)):
        #use laplace to calculate new probabilites, save in list
        #   use sequence before current char to calculate P with laplace and
        #   receive probabilities for whole alphabet as tuple list(char, P)
        previous_sequence = sequence[0:i]

        #conditional probabilities for all letters in alphabet for current previous_sequence, given as tuple_list
        current_probabilities = conditional_probabilities(alphabet, previous_sequence)
        #   index of tuple (char, probability) of current char
        idx = [j for j, tupl in enumerate(current_probabilities) if tupl[0] == sequence[i] ][0]
        #idx = next(i for i, (v, *_) in enumerate(current_probabilities) if v == sequence[i])

        #calculate current temp l(char) and sum it to l
        #   calculate the sum of probabilities of characters BEFORE current character
        probability_before_character = probabilities_before_character(sequence[i], current_probabilities)

        #increment l
        l = l + current_total_probability* decimal.Decimal(probability_before_character)

        #   update current probability which is used to muliply with sum of p before char
        #   current_probabilities[i][1] -> by that, we assume, that char c is on 3rd position(index=3)
        current_total_probability = current_total_probability * decimal.Decimal(current_probabilities[idx][1])

        #multiply P(current char) to final product "seq_probability", l+seq_probability = L
        #   index of char in list of current computed cond. P
        seq_probability = seq_probability * decimal.Decimal(current_probabilities[idx][1])

    L = l + decimal.Decimal(seq_probability)
    return (l,L)
    #return "hehe"

# Function returns octal representation
def float_bin(number, places=3):
    # split() separates whole number and decimal
    # part and stores it in two separate variables
    whole, dec = str(number).split(".")

    # Convert both whole number and decimal
    # part from string type to integer type
    whole = int(whole)
    dec = int(dec)

    # Convert the whole number part to it's
    # respective binary form and remove the
    # "0b" from it.
    res = bin(whole).lstrip("0b") + "."

    # Iterate the number of times, we want
    # the number of decimal places to be
    for x in range(places):
        # Multiply the decimal value by 2
        # and separate the whole number part
        # and decimal part
        whole, dec = str((decimal_converter(dec)) * 2).split(".")

        # Convert the decimal part
        # to integer again
        dec = int(dec)

        # Keep adding the integer parts
        # receive to the result variable
        res += whole

    return res

# Function converts the value passed as
# parameter to it's decimal representation
def decimal_converter(num):
    while num > 1:
        num /= 10
    return num

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    decimal.getcontext().prec = 3000
    # letters = string.ascii_lowercase
    # print(''.join(random.choice(letters) for i in range(10)))

    # al = ''.join(alphabet)
    # print(''.join(random.choice(al) for i in range(10)))
    # print(create_sequence(alphabet, 100))
    #
    # type(["d"]) == list
    # print("".count('e'))
    # for a in alphabet:
    #     print(a)
    alphabet = ['a', 'b', 'c', 'd']
    # print(laplace('d', "aaaa", alphabet))
    #print(float_bin(0.35, 3))

    test_sequence = create_sequence(alphabet, 5)
    # sequence = "kristof"
    # print(sequence[0:1])
    # print(sequence[0:0] == "")
    t = [('a', 0.4), ('b', 0.3), ('c', 0.5)]
    # print(t[1][1])
    # print(next(i for i, (v, *_) in enumerate(t) if v == 'a'))
    # print(sum(n for _, n in t))
    # print(probabilities_before_character('c', t))
    #print(conditional_probabilities(alphabet, "a"))

    l, L = encode("aabcd", alphabet)
    # print(t[[i for i, tupl in enumerate(t) if tupl[0] == 'a'][0]])
    # t = dict(t)
    print(l)
    print(L)


