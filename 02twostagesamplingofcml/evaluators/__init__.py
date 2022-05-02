from evaluators.evaluator import Evaluator
from evaluators.ndcg import NDCGEvaluator
from evaluators.map import MAPEvaluator

############ import ############# 
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class EvaluatorFactory(object):
    """
    Evaluator factory.
    """
    @classmethod
    def create_evaluator(cls, train_interactions, test_interactions,
                         metric='ndcg'):
        if metric == 'ndcg':
            eva = NDCGEvaluator(train_interactions, test_interactions)
        elif metric == 'map':
            eva = MAPEvaluator(train_interactions, test_interactions)
        else:
            raise ValueError('does not support evaluator: {}'.format(metric))
        return eva
