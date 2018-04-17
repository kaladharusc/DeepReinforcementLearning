import numpy as np
import random

import loggers as lg

from game import Game, GameState
from model import Residual_CNN

from agent import Agent, User

import config


class Playing:
    def __init__(self, env, run_version, player1version, player2version, EPISODES, logger, turns_until_tau0, goes_first=0):
        self.EPISODES = EPISODES
        self.turns_until_tau0 = turns_until_tau0
        self.logger  = logger
        self.goes_first = goes_first


        if player1version == -1:
            self.player1 = User('player1', env.state_size, env.action_size)
        else:
            self.player1_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                                      config.HIDDEN_CNN_LAYERS)

            if player1version > 0:
                self.player1_network = self.player1_NN.read(env.name, run_version, player1version)
                self.player1_NN.model.set_weights(self.player1_network.get_weights())
            self.player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, self.player1_NN)

        if player2version == -1:
            self.player2 = User('player2', env.state_size, env.action_size)
        else:
            self.player2_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape, env.action_size,
                                      config.HIDDEN_CNN_LAYERS)

            if player2version > 0:
                self.player2_network = self.player2_NN.read(env.name, run_version, player2version)
                self.player2_NN.model.set_weights(self.player2_network.get_weights())
            self.player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, self.player2_NN)

    def play_one_game(self,e):



        self.logger.info('====================')
        self.logger.info('EPISODE %d OF %d', e + 1, self.EPISODES)
        self.logger.info('====================')
        goes_first = self.goes_first
        player1 = self.player1
        logger = self.logger
        player2 = self.player2
        env = self.env
        turns_until_tau0 = self.turns_until_tau0
        memory = self.memory
        scores = self.scores
        sp_scores = self.sp_scores
        points = self.points

        print(str(e + 1) + ' ', end='')

        state = env.reset()

        done = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        if goes_first == 0:
            player1Starts = random.randint(0,1) * 2 - 1
        else:
            player1Starts = goes_first

        if player1Starts == 1:
            players = {1:{"agent": player1, "name":player1.name}
                    , -1: {"agent": player2, "name":player2.name}
                    }
            logger.info(player1.name + ' plays as X')
        else:
            players = {1:{"agent": player2, "name":player2.name}
                    , -1: {"agent": player1, "name":player1.name}
                    }
            logger.info(player2.name + ' plays as X')
            logger.info('--------------')

        env.gameState.render(logger)

        while done == 0:
            turn = turn + 1

            #### Run the MCTS algo and return an action
            if turn < turns_until_tau0:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1)
            else:
                action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 0)


            if action == "restart":
                break

            if memory != None:
                ####Commit the move to memory
                memory.commit_stmemory(env.identities, state, pi)

            logger.info('action: %d', action)
            for r in range(env.grid_shape[0]):
                logger.info(['----' if x == 0 else '{0:.2f}'.format(np.round(x, 2)) for x in
                             pi[env.grid_shape[1] * r: (env.grid_shape[1] * r + env.grid_shape[1])]])
            # logger.info('MCTS perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(MCTS_value,2))
            # logger.info('NN perceived value for %s: %f', state.pieces[str(state.playerTurn)] ,np.round(NN_value,2))
            logger.info('====================')

            ### Do the action
            state, value, done, _ = env.step(
                action)  # the value of the newState from the POV of the new playerTurn i.e. -1 if the previous player played a winning move
            print("player turn", env.gameState.playerTurn)
            print(env.gameState.board)
            if env.gameState.playerTurn == -1:
                f = open("./communicate/output.txt", "w")
                temp_board = [str(x) for x in env.gameState.board]
                f.write(",".join(temp_board))
                f.close()
            env.gameState.render(logger)

            if done == 1:
                if memory != None:
                    #### If the game is finished, assign the values correctly to the game moves
                    for move in memory.stmemory:
                        if move['playerTurn'] == state.playerTurn:
                            move['value'] = value
                        else:
                            move['value'] = -value

                    memory.commit_ltmemory()

                if value == 1:
                    logger.info('%s WINS!', players[state.playerTurn]['name'])
                    print('%s WINS!' % (players[state.playerTurn]['name']))
                    scores[players[state.playerTurn]['name']] = scores[players[state.playerTurn]['name']] + 1
                    if state.playerTurn == 1:
                        sp_scores['sp'] = sp_scores['sp'] + 1
                    else:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1

                elif value == -1:
                    logger.info('%s WINS!', players[-state.playerTurn]['name'])
                    print('%s WINS!' % (players[-state.playerTurn]['name']))
                    scores[players[-state.playerTurn]['name']] = scores[players[-state.playerTurn]['name']] + 1

                    if state.playerTurn == 1:
                        sp_scores['nsp'] = sp_scores['nsp'] + 1
                    else:
                        sp_scores['sp'] = sp_scores['sp'] + 1

                else:
                    logger.info('DRAW...')
                    print("DRAW")
                    scores['drawn'] = scores['drawn'] + 1
                    sp_scores['drawn'] = sp_scores['drawn'] + 1

                pts = state.score
                points[players[state.playerTurn]['name']].append(pts[0])
                points[players[-state.playerTurn]['name']].append(pts[1])



    def playMatches(self):

        self.env = Game()
        self.scores = {self.player1.name: 0, "drawn": 0, self.player2.name:0}
        self.sp_scores = {'sp': 0, "drawn": 0, 'nsp': 0}
        self.points = {self.player1.name: [], self.player2.name: []}

        for e in range(self.EPISODES):
            self.play_one_game(e)

        return (self.scores, self.memory, self.points, self.sp_scores)

#
# if __name__ == '__main__':
#     env = Game()
#     playing = Playing(env, 1, 1, -1, 10, lg.logger_tourney, 0)
#     playing.playMatches()