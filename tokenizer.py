class Tokenizer():
    def __init__(self):
        self.basic_latin = [chr(c) for c in range(ord(' '), ord('~')+1)]
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.basic_latin)}
        self.idx_to_char = {idx+1: char for idx, char in enumerate(self.basic_latin)}
        self.blank_idx = 0

    def decode(self, log_probs, input_lengths):
        _, bs, _ = log_probs.shape # bs = batch size
        char_probs = log_probs.argmax(dim=2)
        # takes maximum from (timesteps, classes) -> (char index, batch)

        outputs = []
        for n in range(bs):
            encoded_sentence = char_probs[:input_lengths[n],n].tolist()
            decoded_sentence = []
            prev = self.blank_idx
            for encoded_char in encoded_sentence:
                if encoded_char != prev and encoded_char != self.blank_idx:
                    decoded_sentence.append(self.idx_to_char[encoded_char])
                prev = encoded_char
            outputs.append("".join(decoded_sentence))

        return outputs
  
    def encode(self, text):
        return [self.char_to_idx[char] for char in text]