import numpy as np
import nltk
import difflib
import editdistance
import re
from scipy import spatial
import statistics
import string
from nltk.tokenize import sent_tokenize
import gpt_2_simple as gpt2

def preprocess_candidates(candidates):
    for i in range(len(candidates)):
        candidates[i] = candidates[i].strip()
        candidates[i] = '. '.join(candidates[i].split('\n\n'))
        candidates[i] = '. '.join(candidates[i].split('\n'))
        candidates[i] = '.'.join(candidates[i].split('..'))
        candidates[i] = '. '.join(candidates[i].split('.'))
        candidates[i] = '. '.join(candidates[i].split('. . '))
        candidates[i] = '. '.join(candidates[i].split('.  . '))
        while len(candidates[i].split('  ')) > 1:
            candidates[i] = ' '.join(candidates[i].split('  '))
        myre = re.search(r'(\d+)\. (\d+)', candidates[i])
        while myre:
            candidates[i] = 'UNK'.join(candidates[i].split(myre.group()))
            myre = re.search(r'(\d+)\. (\d+)', candidates[i])
        candidates[i] = candidates[i].strip()
    processed_candidates = []
    for candidate_i in candidates:
        sentences = sent_tokenize(candidate_i)
        out_i = []
        for sentence_i in sentences:
            if len(
                    sentence_i.translate(
                        str.maketrans('', '', string.punctuation)).split()
            ) > 1:  # More than one word.
                out_i.append(sentence_i)
        processed_candidates.append(out_i)
    return processed_candidates

def get_redundancy_score(all_summary):

    def if_two_sentence_redundant(a, b):
        """ Determine whether there is redundancy between two sentences. """
        if a == b:
            return 4
        if (a in b) or (b in a):
            return 4
        flag_num = 0
        a_split = a.split()
        b_split = b.split()
        if max(len(a_split), len(b_split)) >= 5:
            longest_common_substring = difflib.SequenceMatcher(
                None, a, b).find_longest_match(0, len(a), 0, len(b))
            LCS_string_length = longest_common_substring.size
            if LCS_string_length > 0.8 * min(len(a), len(b)):
                flag_num += 1
            LCS_word_length = len(a[longest_common_substring[0]:(
                longest_common_substring[0] +
                LCS_string_length)].strip().split())
            if LCS_word_length > 0.8 * min(len(a_split), len(b_split)):
                flag_num += 1
            edit_distance = editdistance.eval(a, b)
            if edit_distance < 0.6 * max(
                    len(a), len(b)
            ):  # Number of modifications from the longer sentence is too small.
                flag_num += 1
            number_of_common_word = len([x for x in a_split if x in b_split])
            if number_of_common_word > 0.8 * min(len(a_split), len(b_split)):
                flag_num += 1
        return flag_num

    redundancy_score = [0.0 for x in range(len(all_summary))]
    for i in range(len(all_summary)):
        flag = 0
        summary = all_summary[i]
        if len(summary) == 1:
            continue
        for j in range(len(summary) - 1):  # for pairwise redundancy
            for k in range(j + 1, len(summary)):
                flag += if_two_sentence_redundant(summary[j].strip(),
                                                  summary[k].strip())
        redundancy_score[i] += -0.1 * flag
    return redundancy_score

def get_similarity_score(candidates, model):
  
  enc_cands = []
  for cand in candidates:
    curr_emb = []
    for sentence in cand:
      curr_emb.append(model.encode(sentence))
    enc_cands.append(curr_emb)

  scores = []
  for i in range(len(candidates)):
    cand_score = []
    for j in range(len(enc_cands[i][:-1])):
      cand_score.append(1 - spatial.distance.cosine(enc_cands[i][j], enc_cands[i][j+1]))
    if len(cand_score) > 0:
      scores.append(statistics.mean(cand_score))
    else:
      scores.append(0)

  return scores

def get_title_score(candidates, title, model):

  scores = []

  embeddings = model.encode(candidates)
  emb_title = model.encode(title)
  for emb in embeddings:
      result = 1 - spatial.distance.cosine(emb_title, emb)
      scores.append(result)

  return scores

def calculate_score(candidates,title,genre,model):

  processed_candidates = preprocess_candidates(candidates)

  redundancy_score = get_redundancy_score(processed_candidates)
  similarity_score = get_similarity_score(processed_candidates,model)
  title_score = get_title_score(candidates,title,model)
  genre_score = get_title_score(candidates,genre,model)

  total_score = []
  for i in range(len(similarity_score)):
    total_score.append(title_score[i] + genre_score[i] + similarity_score[i]/2 + 3*redundancy_score[i])

  '''for i in range(len(candidates)):
    print(candidates[i])
    print(f'Similarity: {similarity_score[i]/2}')
    print(f'Title: {title_score[i]}')
    print(f'Genre: {genre_score[i]}')
    print(f'Redundancy: {redundancy_score[i]}')
    print(f'Total: {total_score[i]}')
    print('\n')'''
  
  return total_score

def samples_selector(samples,title,genre,model):
  
  lens = [0]*len(samples)
  for i in range(len(samples)):
    lens[i] = len(samples[i].split(' '))

  for i in range(2):
    max_len = np.argmax(lens)
    samples.pop(max_len)
    lens.pop(max_len)

    min_len = np.argmin(lens)
    samples.pop(min_len)
    lens.pop(min_len)
  
  #Passar titulo, genero e model aqui
  scores = calculate_score(samples,title,genre,model)
  
  return samples[np.argmax(scores)]
