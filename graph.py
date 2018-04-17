import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111)

def animate(i):
    print("graph")
    graph_data = open("./graphs/train_overall_loss.txt","r").read()


    ax1.clear()
    ax1.plot([float(x) for x in graph_data.split(",")], 'k')
    ax1.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

ani = animation.FuncAnimation(fig,animate,interval=2000)
plt.show()