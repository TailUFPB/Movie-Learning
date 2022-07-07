from utils import *
from sentence_transformers import SentenceTransformer

class Synopsis_Generator:

    def __init__(self):
        self.pos_proc_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        nltk.download('punkt', quiet=True)

    def get_synopsis(self, title, genre, beginning, temperature):

        '''prefix = f"<|startoftext|>~`{title}~^{genre}~@{beginning}"
        temperature = temperature

        samples = gpt2.generate(sess,
              length=150,
              temperature=temperature,
              prefix=prefix,
              truncate="<|endoftext|>",
              nsamples=10,
              batch_size=5,
              return_as_list=True
              )'''
        
        return samples_selector([samp.split('@')[1][:-1] for samp in samples],title,genre,self.pos_proc_model)
        

synp = Synopsis_Generator()
print(synp.get_synopsis('The adventures of Nicolas Cage','action','','d'))