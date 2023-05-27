import pytest 
from src.components.utils import prediciton 

def test_prediction(open,heigh,low,close,volume):
    assert prediciton(open,heigh,low,close,volume) != None 
    


#test project structure 

#test data injection 

#test model training 

#test model evaluation

#test model prediction 