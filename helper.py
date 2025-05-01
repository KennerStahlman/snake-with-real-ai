import matplotlib.pyplot as plt
from IPython import display

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))  # Only create once, globally

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(fig)
    ax.clear()
    ax.set_title('Training...')
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Score')
    ax.plot(scores, label='Score')
    ax.plot(mean_scores, label='Mean')
    ax.set_ylim(ymin=0)
    ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    ax.legend()
    plt.pause(0.1)
