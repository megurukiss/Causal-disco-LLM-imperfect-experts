import numpy as np
import openai
import pickle
import random

from scipy.special import softmax
from scipy.optimize import fsolve
from utils.dspy_utils import LLMExpert

PROMPT_TEMPLATE = """
  Among these two options which one is the most likely true:
  (A) {0} {2} {1}
  (B) {1} {2} {0}
  The answer is: 
"""

DSPY_PROMPT_TEMPLATE = """
{0} {2} {1}
"""

OPTIONS = ['(A)', '(B)']

LOCK_TOKEN = ' ('

VERBS = ["provokes", " triggers","causes", "leads to", "induces", "results in", "brings about", "yields", "generates", "initiates", "produces", "stimulates", "instigates", "fosters", "engenders", "promotes", "catalyzes", "gives rise to", "spurs", "sparks"]

llm_expert = LLMExpert()

def get_prompt(edge, codebook, verb=None):

  node_i, node_j = edge
  long_name_node_i = codebook.loc[codebook['var_name']==node_i, 'var_description'].to_string(index=False)
  long_name_node_j = codebook.loc[codebook['var_name']==node_j, 'var_description'].to_string(index=False)

  if 'Series' in long_name_node_i:
    print(f"{node_i} is not defined")
  if 'Series' in long_name_node_j:
    print(f"{node_j} is not defined")
  
  if verb is None:
    verb = random.choice(VERBS)

  options = PROMPT_TEMPLATE.format(long_name_node_i, long_name_node_j, verb)

  return options

def get_dspy_prompt(edge, codebook, verb=None):
   
  node_i, node_j = edge
  long_name_node_i = codebook.loc[codebook['var_name']==node_i, 'var_description'].to_string(index=False)
  long_name_node_j = codebook.loc[codebook['var_name']==node_j, 'var_description'].to_string(index=False)
  if 'Series' in long_name_node_i:
    print(f"{node_i} is not defined")
  if 'Series' in long_name_node_j:
    print(f"{node_j} is not defined")
  
  if verb is None:
    verb = random.choice(VERBS)
  
  prompts=[]
  # A -> B
  prompt1 = DSPY_PROMPT_TEMPLATE.format(long_name_node_i, long_name_node_j, verb)
  # B -> A
  prompt2 = DSPY_PROMPT_TEMPLATE.format(long_name_node_j, long_name_node_i, verb)
  prompts.append(prompt1)
  prompts.append(prompt2)

  return prompts

def get_lms_probs(undirected_edges, codebook, tmp_scaling=1, engine='davinci-002', model="dspy"):
  """
  return: dictionary of tuple and their likelihood of being wrong by the LM
  example {('Age', 'Disease'): 0.05, ...}
  """

  probs = {}
  decisions = []

  for edge in undirected_edges:
      if model == "dspy":
        responses=gpt_call(edge, codebook)
        # scale the probs to be between 0 and 1
        logits = np.log(np.array(responses) / (1 - np.array(responses) + 1e-10))
        responses = softmax(logits / tmp_scaling)
        probs[(edge[0], edge[1])] = responses[0]
        probs[(edge[1], edge[0])] = responses[1]
        
        if responses[0] > responses[1]:
          decisions.append((edge[0], edge[1]))
        else:
          decisions.append((edge[1], edge[0]))

      else:
        log_scores = gpt3_scoring(edge, codebook, options=OPTIONS, lock_token=LOCK_TOKEN, engine=engine)
        scores = softmax(log_scores / tmp_scaling)

        
        probs[(edge[0], edge[1])] = scores[0]
        probs[(edge[1], edge[0])] = scores[1]

        if scores[0] > scores[1]:
          decisions.append((edge[0], edge[1]))
        else:
          decisions.append((edge[1], edge[0]))

  return probs, decisions

def temperature_scaling(directed_edges, codebook, engine):
  err_scores = []
  num_errs = 0

  for edge in directed_edges:
      # node_i -> node_j 
      options = get_prompt(edge, codebook)

      log_scores = gpt3_scoring(options, options=OPTIONS, lock_token=LOCK_TOKEN, engine=engine)

      if log_scores[0] < log_scores[1]:
        num_errs += 1
        err_scores.append(log_scores[1])
        print(edge)
      else:
        err_scores.append(log_scores[0]) 

  estimated_error = num_errs / len(directed_edges)
  err_scores = np.array(err_scores)

  equation = lambda t: np.average(np.exp(err_scores / t) / (np.exp(err_scores / t) + np.exp((1 - err_scores) / t))) - estimated_error

  temperature = fsolve(equation, 1.)
  print(np.average(np.exp(err_scores / temperature) / (np.exp(err_scores / temperature) + np.exp((1 - err_scores) / temperature))))

  return float(temperature), estimated_error

def gpt_call(edge, codebook, cache_file='dspy_cache.pickle',verb="causes"):
  DSPy_CACHE = {}
  try:
      with open(cache_file, 'rb') as f:
          DSPy_CACHE = pickle.load(f)
  except:
      pass
  responses=[]

  new_call = False
  prompts = get_dspy_prompt(edge, codebook, verb)
  for prompt in prompts:
     # check if prompt is in cache
    if prompt in DSPy_CACHE.keys():
      prob = DSPy_CACHE[prompt]
      responses.append(prob)
    else:
      prob = llm_expert(prompt).probability
      responses.append(prob)
      DSPy_CACHE[prompt] = prob
      new_call = True

  if new_call:
    with open(cache_file, 'wb') as f:
      pickle.dump(DSPy_CACHE, f)

  return np.array(responses)


def gpt3_call(engine, edge, codebook, options, max_tokens=128, temperature=0, 
              logprobs=1, echo=False, cache_file='llm_cache.pickle'):
  cache_file = engine + '_llm_cache.pickle'
  LLM_CACHE = {}
  try:
      with open(cache_file, 'rb') as f:
          LLM_CACHE = pickle.load(f)
  except:
      pass
  
  verbs = random.sample(VERBS, len(VERBS))
  for verb in verbs:
    prompt = get_prompt(edge, codebook, verb)
    gpt3_prompt_options = [f"{prompt}{o}" for o in options]

    full_query = ""
    for p in gpt3_prompt_options:
      full_query += p

    id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
    if id in LLM_CACHE.keys():
      response = LLM_CACHE[id] 
      break
  
  # if ID is not in pickle (with any verb option)
  else:
    print('no cache hit, api call')
    response = openai.Completion.create(engine=engine, 
                                        prompt=gpt3_prompt_options, 
                                        max_tokens=max_tokens, 
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo)
    LLM_CACHE[id] = response
    with open(cache_file, 'wb') as f:
      pickle.dump(LLM_CACHE, f)
  return response


def gpt3_scoring(edge, codebook, options, engine="text-davinci-002", verbose=False, n_tokens_score=9999999999, lock_token=None, ): 
    verbose and print("Scoring", len(options), "options") 
     
    response = gpt3_call(engine, edge, codebook, options, max_tokens=0, logprobs=1, temperature=0, echo=True, ) 
    scores = [] 
    for option, choice in zip(options, response["choices"]): 
        if lock_token is not None: 
            n_tokens_score = choice["logprobs"]["tokens"][::-1].index(lock_token)
        tokens = choice["logprobs"]["tokens"][-n_tokens_score:] 
        verbose and print("Tokens:", tokens) 
        token_logprobs = choice["logprobs"]["token_logprobs"][-n_tokens_score:] 
        total_logprob = 0 
        denom = 0 
        for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)): 
            if token_logprob is not None: 
                denom += 1 
                total_logprob += token_logprob 
        scores.append(total_logprob) 
    return np.array(scores)


if __name__ == '__main__':
  options = """
            Options:
            (A) Cancer causes smoking
            (B) Smoking causes cancer
            The answer is: 
            """
  log_scores = gpt3_scoring(options, options=['(A)', '(B)'], lock_token=' (')
  scores = softmax(log_scores)
  print(scores)