import threading
import berserk
import chess

from src.engines.neural import NeuralInterface

TOKEN = '...'

USER_TO_CHALLENGE = '...'

class Game(threading.Thread):
    def __init__(self, client, game_event, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_event["game"]["gameId"]
        self.color = game_event["game"]["color"] == 'white'
        self.client = client
        self.stream = client.bots.stream_game_state(self.game_id )
        self.current_state = next(self.stream)

        self.engine = NeuralInterface()


    def run(self):
        for event in self.stream:
            print(event)
            if event['type'] == 'gameState':
                self.handle_state_change(event)
            elif event['type'] == 'gameFull':
                self.handle_state_change(event['state'])
            elif event['type'] == 'chatLine':
                self.handle_chat_line(event)
            elif event['type'] == 'gameFinish':
                break


    def handle_state_change(self, game_state):
        move_list = game_state['moves'].split(' ')

        board = chess.Board()
        for move in move_list:
            board.push(chess.Move.from_uci(move))
        
        if self.color == board.turn:
            engine_move = self.engine.play(board)
            self.client.bots.make_move(self.game_id, engine_move.uci())


    def handle_chat_line(self, chat_line):
        print(f"Chat: {chat_line}")

if __name__ == '__main__':
    session = berserk.TokenSession(TOKEN)
    client = berserk.Client(session)

    client.challenges.create(username=USER_TO_CHALLENGE, clock_limit= 10 * 60, clock_increment = 5, rated=False)

    for event in client.bots.stream_incoming_events():
        if event['type'] == 'challenge':
            client.bots.decline_challenge(event['challenge']['id'])

        elif event['type'] == 'gameStart':
            game = Game(client, event)
            game.start()

