import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot(acc, k):
    plt.plot(k, acc)
    
    plt.title("BM25 Top K Accuracy")
    plt.xlabel("K")
    plt.ylabel("Accuracy");

    #fig.savefig("accuracy.png")
    plt.show()
	
def get_top_kth_acc(guesses, k):
    correct = 0
    for ans, guess in guesses:
        for g in guess[:k]:
            if g[0] == ans:
                correct += 1
                break
    return correct/len(guesses)
	
	
with open("guesses_test", 'rb') as f:
    guesses = pickle.load(f)
    # answer, List[Tuple[str, float]]
k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
acc_list = []
for k in k_list:
    acc_list.append(get_top_kth_acc(guesses, k))
plot(acc_list, k_list)