import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import seaborn as sn


def plot_confusion_matrix(confusion_matrix,labels,label):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(label)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(label)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(),
             rotation_mode="anchor")

    for i in range(len(labels)):
        for j in range(len(label)):
            text = ax.text(j, i, confusion_matrix[i, j],
                           ha="center", va="center", color="black")

    # ax.set_title("confusion matrix of cnn")
    fig.tight_layout()
    plt.show()

def main():
    confusion_matrix = np.array([[0.2775, 0.3144, 0.2946],
                                [0.25,  0.257, 0.253],
                                [0.403, 0.359, 0.38],
                                [0.317,0.317,0.317]])

    labels = ('group', 'headshot', 'not_person','total')
    label = ('Precision', 'Recall','F1-score')
    plot_confusion_matrix(confusion_matrix,labels,label)

if __name__ == '__main__':
    main()