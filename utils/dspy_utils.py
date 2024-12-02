import dspy
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='./../.env')
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'),temperature=0.9)
dspy.configure(lm=lm)
colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

class CausalRelation(dspy.Signature):
    """ Given a probable causal relation between two nodes, output the probability of the relation."""
    context: list[str] = dspy.InputField(desc="May contain relevant facts")
    causal_relation: str = dspy.InputField(desc="The causal relation between two nodes in a Markov equivalence class. Format: 'A triggers B', means A is one of the causes of B.")
    probability: float = dspy.OutputField(desc="The probability (between 0 and 1) that the causal relation is true.")

class LLMExpert(dspy.Module):
    def __init__(self):
        self.causal_relation=dspy.ChainOfThought(CausalRelation)
        # retrieve the wiki knowledge based on the given causal relation
        self.retriever = dspy.Retrieve(k=3)
    
    def forward(self, causal_relation: str):
        context=self.retriever(causal_relation).passages
        response=self.causal_relation(context=context, causal_relation=causal_relation)
        return response

if __name__ == '__main__':
    expert = LLMExpert()
    llm_expert = LLMExpert()
    response=llm_expert(causal_relation="age triggers socioeconomic status")
    print(response.probability)