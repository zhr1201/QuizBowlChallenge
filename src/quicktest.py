from qanta.guesser import Guesser
from qanta.model_proxy import ModelProxy
from qanta.dataset import QuizBowlDataset
from tqdm import tqdm
import numpy as np
import json

k=10
config_file = 'conf/BM25-SpacyNERReranker.yaml'
config_file = 'conf/BM25-SpacyNERNNReranker.yaml'
config_file = 'conf/BM25PRE-Retriever.yaml'
config_file = 'conf/BM25-SpacyNERReranker.yaml'
config_file = 'conf/BM25-SpacyNERQUESReranker.yaml'
config_file = 'conf/BM25-SpacyNERQUESReranker.yaml'
config_file = 'conf/BM25-SpacyNERNNv2Reranker.yaml'
config_file = 'conf/BM25-SpacyNERNNQEReranker.yaml'
config_file = 'conf/BM25-SpacyNERQUESTAGReranker.yaml'
config_file = 'conf/BM25-SpacyNERQUESReranker.yaml'
##config_file = 'conf/BM25-SpacyNERWIKITAGReranker.yaml'
#model = ModelProxy.load(config_file)
dataset = QuizBowlDataset(guesser_train=True,buzzer_train=True)
test_questions  = [' '.join(q) for q in dataset.test_data()[0]]
test_answers = dataset.test_data()[1]
#test_guesses = []
dev_questions  = [' '.join(q) for q in dataset.dev_data()[0]]
dev_answers = dataset.dev_data()[1]
with open('data/qanta.tacl-trick.json','r') as f:
    advset = json.load(f)
adv_questions = [x['text'] for x in advset['questions']]
adv_answers = [x['text'] for x in advset['page']]
#dev_guesses = []


def evaluate(questions,answers,k,config_file):
    config_file = config_file
    model = ModelProxy.load(config_file)
    guesses = []
    for i in tqdm(range(len(questions))):
        # if i%100 == 0:
        #     print(f"{i}, {answers[i]}, {questions[i]}\n")
        #     print(f"{model.guess(questions[i],k)[0]}")
        guesses.append((answers[i], model.guess(questions[i],k)[0]))
    topk = [[y[0] for y in x[1] ]for x in guesses]
    top1 =[x[1][0][0] for x in  guesses]
    endacc = np.sum(np.array(top1)==np.array(answers))/len(questions)
    count =0
    for i,ans in enumerate(answers):
        if ans in topk[i]:
            count+=1
    topkacc = count/len(answers)
    return endacc, topkacc

def evaluate2(questions,answers,k,config_file):
    config_file = config_file
    model = ModelProxy.load(config_file)
    guesses = []
    # for i in tqdm(range(len(questions))):
    #     # if i%100 == 0:
    #     #     print(f"{i}, {answers[i]}, {questions[i]}\n")
    #     #     print(f"{model.guess(questions[i],k)[0]}")
    guesses = model.guess(questions,k)
    topk = [[y[0] for y in x ]for x in guesses]
    #topk = [[y[0] for y in x[1] ]for x in guesses]
    top1 =[x[0][0] for x in  guesses]
    endacc = np.sum(np.array(top1)==np.array(answers))/len(questions)
    count =0
    for i,ans in enumerate(answers):
        if ans in topk[i]:
            count+=1
    topkacc = count/len(answers)
    return endacc, topkacc
# print(config_file+"\n")
# endacc, topkacc = evaluate2(dev_questions,dev_answers,10,config_file)
# print(f"end acc: {endacc} topk acc: {topkacc}")

# endacc, topkacc =evaluate2(test_questions,test_answers,10,config_file)
# print(f"end acc: {endacc} topk acc: {topkacc}")

endacc, topkacc =evaluate2(adv_questions,adv_answers,10,config_file)
print(f"end acc: {endacc} topk acc: {topkacc}")
# for i in tqdm(range(len(test_questions))):
#     if i%100 == 0:
#         print(f"{i}, {test_answers[i]}, {test_questions[i]}\n")
#         print(f"{model.guess(test_questions[i],k)[0]}")
#     test_guesses.append((test_answers[i], model.guess(test_questions[i],k)[0]))


# for i in tqdm(range(len(dev_questions))):
#     if i%100 == 0:
#         print(f"{i}, {dev_answers[i]}, {dev_questions[i]}\n")
#         print(f"{model.guess(dev_questions[i],k)[0]}")
#     dev_guesses.append((test_answers[i], model.guess(dev_questions[i],k)[0]))

# dev_topk = [[y[0] for y in x[1] ]for x in dev_guesses]
# dev_top1 =[x[1][0][0] for x in  dev_guesses]
# test_topk = [[y[0] for y in x[1] ]for x in test_guesses]
# test_top1 = [x[1][0][0] for x in test_guesses]

# endcc = np.sum(np.array(dev_top1)==np.array(dev_answers))/len(dev_questions)
# count =0
# for ans in dev_answers:
#     if ans in dev_topk:
#     count+=1
# topkacc = count/len(dev_answers)
# test_guesses = model.guess(test_questions,10)
# dev_guesses = model.guess(dev_questions,10)
