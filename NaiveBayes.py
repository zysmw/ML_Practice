from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, CategoricalNB, ComplementNB
import numpy as np

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
test = np.array([[-0.8, -1]])

GNB = GaussianNB()
# MNB = MultinomialNB()
# CNB = CategoricalNB()

GNB.fit(X, Y)
# MNB.fit(X, Y)
# CNB.fit(X, Y)

print(GNB.predict(test))
print(GNB.predict_proba(test))
print(GNB.predict_log_proba(test))
print("\n")
# print(MNB.predict(test))
# print(MNB.predict_proba(test))
# print(MNB.predict_log_proba(test))
print("\n")
# print(CNB.predict(test))
# print(CNB.predict_proba(test))
# print(CNB.predict_log_proba(test))

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 1, 0.01)
Entropy = np.array([])
Gini = np.array([])
for item in x:
    if item == 0:
        e = 0
        g = 0
    else:
        e = -item * np.log(item) - (1 - item) * np.log(1 - item)
        g = 2 * item * (1 - item)
    Entropy = np.append(Entropy, e/2)
    Gini = np.append(Gini, g)

fig, ax = plt.subplots(1, 1)
plt.plot(x, Entropy, "b", label="Half Entropy")
plt.plot(x, Gini, "r", label="Gini Index")
plt.title("Entropy--Gini")
plt.xlabel("P")
plt.legend()
plt.grid()
plt.show()
