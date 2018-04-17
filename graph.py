import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
#
def animate(i):
    print("graph")
    graph_data = open("./graphs/train_overall_loss.txt","r").read()


    ax1.clear()
    ax1.plot([float(x) for x in graph_data.split(",")], 'k')
    ax1.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='upper right')
    ax1.set_xlabel("train_overall_loss")

    graph_data = open("./graphs/policy_head_loss.txt", "r").read()
    ax2.clear()
    ax2.plot([float(x) for x in graph_data.split(",")], 'k')
    ax2.set_xlabel("policy_head_loss")
    # ax1.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

    graph_data = open("./graphs/value_head_loss.txt", "r").read()
    ax3.clear()
    ax3.plot([float(x) for x in graph_data.split(",")], 'k')
    ax3.set_xlabel("value_head_loss")
    # ax1.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

ani = animation.FuncAnimation(fig,animate,interval=2000)
plt.show()