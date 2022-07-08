from utils import *
from sentence_transformers import SentenceTransformer
import gpt_2_simple as gpt2

class Synopsis_Generator:

    def __init__(self):
        self.pos_proc_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        nltk.download('punkt', quiet=True)
        self.sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(self.sess)

    def get_synopsis(self, title, genre, beginning, temperature):
        
        prefix = f"<|startoftext|>~`{title}~^{genre}~@{beginning}"
        samples = gpt2.generate(self.sess,
              length=200,
              temperature=temperature,
              prefix=prefix,
              truncate="<|endoftext|>",
              nsamples=10,
              batch_size=5,
              return_as_list=True
              )
        
        return samples_selector([samp.split('@')[1][:-1] for samp in samples],title,genre,self.pos_proc_model)

if __name__ == "__main__":
    synp = Synopsis_Generator()
    print(synp.get_synopsis('Death on the Moon','mystery','',0.7))
    print(synp.get_synopsis('The spaghetti war','documentary','',0.7))
