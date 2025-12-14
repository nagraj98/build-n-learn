# Implementing Byte pair encodings (BPE) for embeddings

import re
import os
import json


class BPEEncoder:
    """
    Docstring for BPEEncoder
    """
    def __init__(self, data_directory: str, output_directory: str):
        self.data_directory = data_directory
        self.output_directory = output_directory

        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Data directory {data_directory} does not exist.")
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        self.content = self.build_content(data_directory)

        self.vocabulary = None
        self.vocabulary_map = None

        self.OUTPUT_VOCABULARY_FILE = os.path.join(output_directory, "vocabulary.txt")
        self.OUTPUT_VOCABULARY_MAP_FILE = os.path.join(output_directory, "vocabulary_map.json")


    def build_content(self, data_directory: str):
        """
        Docstring for build_data
        
        :param data_directory: Description
        :type data_directory: str
        """

        data_files = [os.path.join(data_directory, file)
                      for file in os.listdir(data_directory) if file.endswith('.txt')]

        try:
            content = ""
            for data_file in data_files:
                # read the file
                with open(data_file, 'r', encoding='utf-8') as file:
                    content += file.read()
        except FileNotFoundError as fnf_error:
            print(f"FileNotFoundError : {fnf_error}")

        # clean the file - remove extra whitespaces (\t and \n)
        content = re.sub(r'\t+', '\t', content)
        content = re.sub(r'\n+', '\n', content)

        return content

    def build_vocabulary(self, vocab_size = 1000, frequency_threshold: int = 20):
        """
        Docstring for build_vocabulary
        
        :param content: Description
        :param vocab_size: Description
        :param frequency_threshold: Description
        :type frequency_threshold: int
        """

        content_chars = list(self.content)

        # initialise vocabulary with all characters from the content
        # down the line this will hold the 
        self.vocabulary = list(sorted(set(content_chars)))
        self.vocabulary_map = {}

        itr = 0
        max_frequency = frequency_threshold + 1
        pair_map = {}
        while((len(self.vocabulary) < vocab_size) and max_frequency > frequency_threshold):

            # build the pair map with frequencies
            pair_map = {}
            for i in range(len(content_chars) - 1):
                pair = (content_chars[i], content_chars[i+1])
                if pair in pair_map:
                    pair_map[pair] += 1
                else:
                    pair_map[pair] = 1

            # get the most frequent pairs as tokens
            max_frequency = max(pair_map.values())
            token_pairs = [k for k,v in pair_map.items() if v == max_frequency ]
            tokens = ["".join(tp) for tp in token_pairs]
            self.vocabulary += tokens

            # vocabulary map should have the token pair, and the resulting token as a map
            # self.vocabulary_map += {pair: "".join(pair) for pair in token_pairs}
            self.vocabulary_map.update({pair: "".join(pair) for pair in token_pairs})

            # merge the most frequent pairs in the content
            i = 0
            new_content_chars = []
            while(i < (len(content_chars)-1)):
                pair = (content_chars[i], content_chars[i+1])
                if pair in token_pairs:
                    new_content_chars.append("".join(pair))
                    i += 2
                else:
                    new_content_chars.append(content_chars[i])
                    i += 1

            content_chars = new_content_chars
            itr+=1

        print(f"iteration {itr}, map_length is {len(pair_map)}, current max_frequency is {max_frequency}, length of vocabulary is {len(self.vocabulary)}")

        #save the self.vocabulary to a file
        with open(self.OUTPUT_VOCABULARY_FILE, 'w', encoding='utf-8') as vocab_file:
            for token in self.vocabulary:
                vocab_file.write(token + '\n')

        # store the vocabulary map to a json file
        # making the vocab jsonsafe by concatenating the tuple values with a special character in between like 
        json_safe_vocab = {
            "\u241F".join(k): v for k, v in self.vocabulary_map.items()
        }
        with open(self.OUTPUT_VOCABULARY_MAP_FILE, 'w', encoding='utf-8') as vocab_map_file:
            json.dump(json_safe_vocab, vocab_map_file, ensure_ascii=False, indent=4)

        print(f"Vocabulary and vocabulary map saved to {self.output_directory}")

my = {"hey":3}
my.__str__()



class Tokenizer:

    def __init__(self, input_text: str, vocabulary_map: dict):
        self.input_text = input_text
        self.vocabulary_map = vocabulary_map
    
    def encode_input(self):
        """
        Docstring for encode_input
        tokenize the input text using the vocabulary map
        
        :param input_text: Description
        :type input_text: str
        :param vocabulary_map: Description
        :type vocabulary_map: dict
        """

        input_chars = list(self.input_text)
        
        for mapping, token in self.vocabulary_map.items():
            encoded_tokens = []
            pair_length = len(token)

            # loop over the input chars to find the mapping and replace with token
            i=0
            while(i < (len(input_chars) - (pair_length - 1))):
                pair = tuple(input_chars[i:i+pair_length])
                if pair == mapping:
                    encoded_tokens.append(token)
                    i += pair_length
                else:
                    encoded_tokens.append(input_chars[i])
                    i += 1

        return encoded_tokens



def main():

    INPUT_DATA_DIRECTORY = "./data/"
    OUTPUT_DIRECTORY = "./vocab/"

    BPEEncoder_instance = BPEEncoder(data_directory=INPUT_DATA_DIRECTORY, output_directory=OUTPUT_DIRECTORY)
    BPEEncoder_instance.build_vocabulary(vocab_size=200, frequency_threshold=200)


if __name__ == "__main__":
    main()
