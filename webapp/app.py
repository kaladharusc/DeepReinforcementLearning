from flask import Flask, render_template
from flask_socketio import SocketIO, emit



# import middleware

import time
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app)


@app.route("/")
def hello():

    return render_template('index.html')

@socketio.on('human')
def handle_message(message):
    f = open("../communicate/input.txt",'w')
    f.write(message['data'])
    f.close()
    # f = open("../communicate/output.txt",'r')
    # out = f.readline()
    # while out == "":
    #     out = f.readline()
    # f.close()
    f = open("../communicate/output.txt", 'w')
    f.write("")
    f.close()
    wait()
    # jarivs(out)
    # middleware.writeToInput(message)

@socketio.on('jarvis')
def jarivs(messge):
    emit("jarvis", messge, json=True)

@socketio.on("train")
def wait():
    f = open("../communicate/output.txt",'r')
    out = f.readline().strip()
    f.close()
    print("out")
    print(len(out))
    while len(out) == 0:
        f = open("../communicate/output.txt", 'r')
        out = f.readline()
        print(len(out))
        f.close()
    # f = open("../communicate/output.txt", 'w')
    # f.write("")
    # f.close()
    print(out)
    jarivs(out)


@socketio.on("start")
def start():
    f = open("../communicate/output.txt",'r')
    out = f.readline().strip()
    f.close()
    print("out")

    if out != "":
        # f = open("../communicate/output.txt", 'w')
        # f.write("")
        # f.close()
        jarivs(out)

@socketio.on("progress")
def graph():
    f = open("../data/progress.txt")
    data = f.read()
    f.close()
    emit("progress", data, json=True)

if __name__ == '__main__':
    socketio.run(app)