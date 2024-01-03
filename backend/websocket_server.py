from flask import Flask, render_template
from flask_socketio import SocketIO, emit

websocket_server = Flask(__name__, template_folder="../frontend/public")
socketio = SocketIO(websocket_server)


@websocket_server.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('live_data')
def handle_live_data(data):
    prediction = 3.141527  # Replace with your neural network logic
    emit('prediction', prediction)


if __name__ == '__main__':
    socketio.run(websocket_server, debug=True)
