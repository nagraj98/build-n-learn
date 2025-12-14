# Implementing Byte pair encodings (BPE) for embeddings

import re

def build_vocabulary(data_file):

    # read the file
    try:
        with open(data_file, 'r') as file:
            content = file.read()
    except Exception as e:
        print(f"An Exception occured : {e}")

    # clean the file - remove extra whitespaces (\t and \n)
    content = re.sub(r'\t+', '\t', content)
    content = re.sub(r'\n+', '\n', content)
    content_chars = list(content)

    print(content_chars[:100])

    vocabulary = ["highest_frequency_elements"]

    itr = 0
    max_value = 600
    while((len(vocabulary) < 1000) and max_value > 100):
    # while(itr < 100):

    # input_tokens = ["He",  "l",  "l" o this", "is", "Harry", "calling", "from", "Hogwarts"]

        pair_map = {}
        for i in range(len(content_chars) - 1):
            # pair = content_chars[i] + content_chars[i+1]
            pair = (content_chars[i], content_chars[i+1])
            if pair in pair_map:
                pair_map[pair] += 1
            else:
                pair_map[pair] = 1

        max_value = max(pair_map.values())
        token_pairs = [k for k,v in pair_map.items() if v == max_value ]
        tokens = ["".join(tp) for tp in token_pairs]
        vocabulary += tokens

        new_content_chars = []

        i = 0
        while(i < (len(content_chars)-1)):
            pair = (content_chars[i], content_chars[i+1])
            if pair in token_pairs:
                new_content_chars.append("".join(pair))
                i += 2
            else:
                new_content_chars.append(content_chars[i])
                i += 1

        content_chars = new_content_chars

        # print()
        print(vocabulary)

        print(content_chars[:100])

        itr+=1
        # break
    # print(f"iteration {itr}, map_length is {len(pair_map)}, max_value is {max_value}, length of vocabulary is {len(vocabulary)}")
    print(vocabulary)


INPUT_FILE_PATH = "./data/The_Complete_Works_of_Swami_Vivekananda.txt"
    
build_vocabulary(INPUT_FILE_PATH)


# TODO : 
# 1. add characters to vocabulary initially
# 2. 