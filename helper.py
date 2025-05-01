import matplotlib.pyplot as plt
from IPython import display

fig, ax = plt.subplots()

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    ax.clear()
    ax.plot(scores, label='Score')
    ax.plot(mean_scores, label='Mean')
    ax.legend()
    ax.set_title("Training Progress")
    ax.set_xlabel("Games")
    ax.set_ylabel("Score")
    ax.grid(True)
    display.display(fig)
    plt.pause(0.001)
