import numpy as np
import logging


class Game:

    def __init__(self):
        self.currentPlayer = 1
        self.gameState = GameState(np.array([
            0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            -1, 0, -1, 0, -1, 0, -1, 0,
            0, -1, 0, -1, 0, -1, 0, -1,
            -1, 0, -1, 0, -1, 0, -1, 0
        ], dtype=np.int), 1)
        self.actionSpace = np.array([
            0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            -1, 0, -1, 0, -1, 0, -1, 0,
            0, -1, 0, -1, 0, -1, 0, -1,
            -1, 0, -1, 0, -1, 0, -1, 0
        ], dtype=np.int)
        self.pieces = {'2': 'B', '1': 'b', '0': '-', '-1': 'w', '-2': 'W'}
        self.grid_shape = (8, 8)
        self.input_shape = (2, 8, 8)
        self.name = 'checkers'
        self.state_size = len(self.gameState.binary)
        self.action_size = len(self.actionSpace)

    def reset(self):
        self.gameState = GameState(np.array([
            0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            -1, 0, -1, 0, -1, 0, -1, 0,
            0, -1, 0, -1, 0, -1, 0, -1,
            -1, 0, -1, 0, -1, 0, -1, 0
        ], dtype=np.int), 1)
        self.currentPlayer = 1
        return self.gameState

    def step(self, action):
        next_state, value, done = self.gameState.takeAction(action)
        self.gameState = next_state
        self.currentPlayer = -self.currentPlayer
        info = None
        return ((next_state, value, done, info))

    def identities(self, state, actionValues):
        identities = [(state, actionValues)]

        currentBoard = state.board
        currentAV = actionValues

        # TODO: check if someething goes wrong
        currentBoard = np.array(currentBoard)
        currentAV = np.array(currentAV)

        identities.append((GameState(currentBoard, state.playerTurn), currentAV))

        return identities


class GameState():
    def __init__(self, board, playerTurn):
        self.board = board
        self.pieces = {'2': 'B', '1': 'b', '0': '-', '-1': 'w', '-2': 'W'}
        self.gamesWithoutKillOrKing = 0
        # self.winners = []
        self.playerTurn = playerTurn
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()

    #
    # def allowedMovesDiagonally(self, i, no_occupant=False):
    #     allowed = []
    #     # left corner
    #     if (self.playerTurn == -1 and i % 8 == 0):
    #         allowed.append(i - 7)
    #     elif (self.playerTurn == 1 and i % 8 == 0):
    #         allowed.append(i + 9)
    #     # right corner
    #     if (self.playerTurn == -1 and i % 8 == 7):
    #         allowed.append(i - 9)
    #     elif (self.playerTurn == 1 and i % 8 == 7):
    #         allowed.append(i + 7)
    #
    #     # middle men
    #     if (self.playerTurn == -1):
    #         allowed.append(i - 7)
    #         allowed.append(i - 9)
    #     elif (self.playerTurn == 1):
    #         allowed.append(i + 7)
    #         allowed.append(i + 9)
    #
    #
    #     if no_occupant:
    #         for j in allowed:
    #             if self.board[j] != 0:
    #                 allowed.remove(j)
    #     else:
    #         for j in allowed:
    #             if self.board[j] == self.playerTurn:
    #                 allowed.remove(j)
    #     return allowed

    def allowedMovesDiagonallyWithSteps(self, i, steps=1, king=False):
        allowed = []
        # black
        if (self.playerTurn == 1 or king):
            next_row = (i // 8) + steps
            steps_away = i % 8
            perpendiular_position = 8 * next_row + steps_away
            if perpendiular_position % 8 == 0:
                allowed = allowed + [perpendiular_position + (1 * steps)]
            elif perpendiular_position % 8 == 7:
                allowed = allowed + [perpendiular_position - (1 * steps)]
            else:
                allowed = allowed + [perpendiular_position - (1 * steps), perpendiular_position + (1 * steps)]

        # white
        if (self.playerTurn == -1 or king):
            next_row = (i // 8) - steps
            steps_away = i % 8
            perpendiular_position = 8 * next_row + steps_away

            if perpendiular_position % 8 == 0:
                allowed = allowed + [perpendiular_position + (1 * steps)]
            elif perpendiular_position % 8 == 7:
                allowed = allowed + [perpendiular_position - (1 * steps)]
            else:
                allowed = allowed + [perpendiular_position - (1 * steps),
                                     perpendiular_position + (1 * steps)]

        return allowed

    def getKillActions(self, i, king=False):
        allowed = self.allowedMovesDiagonallyWithSteps(i, king=king)
        final_allowed = []
        allowed_len = len(allowed)
        steps = 2
        while len(allowed) != 0:
            if allowed_len == 0:
                allowed_len = len(allowed)
                steps += 2
            j = allowed.pop(0)
            if self.board[j] == -1 * self.playerTurn:
                temp = self.allowedMovesDiagonallyWithSteps(i, steps=steps, king=king)
                temp_current = self.allowedMovesDiagonallyWithSteps(j)
                temp_allowed = [value for value in temp if temp in temp_current]
                if len(temp_allowed) != 0:
                    allowed = allowed + temp_allowed
                elif len(temp_allowed) != 0:
                    final_allowed = final_allowed + [{'from': i, 'to': temp_allowed[0], 'kill': j}]
                allowed_len -= 1
            elif self.board[j] == 0:
                final_allowed = final_allowed + [{'from': i, 'to': j, 'kill': None}]

        return final_allowed

    def _allowedActions(self):
        allowed = []

        for i in range(len(self.board)):
            if self.board[i] == self.playerTurn:
                # kill/jump move (loop)
                allowed = allowed + self.getKillActions(i)

            # TODO: king moves

        # for i in range(len(self.board)):
        #     if i >= len(self.board) - 7:
        #         if self.board[i] == 0:
        #             allowed.append(i)
        #     else:
        #         if self.board[i] == 0 and self.board[i + 7] != 0:
        #             allowed.append(i)
        self.allowedActionsFull = allowed
        temp = []
        for obj in allowed:
            temp.append(obj["to"])

        return temp

    def _binary(self):

        currentplayer_position = np.zeros(len(self.board), dtype=np.int)
        currentplayer_position[self.board == self.playerTurn] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -self.playerTurn] = 1

        position = np.append(currentplayer_position, other_position)

        return (position)

    def _convertStateToId(self):
        player1_position = np.zeros(len(self.board), dtype=np.int)
        player1_position[self.board == 1] = 1

        other_position = np.zeros(len(self.board), dtype=np.int)
        other_position[self.board == -1] = 1

        position = np.append(player1_position, other_position)

        id = ''.join(map(str, position))

        return id

    def _checkForEndGame(self):
        if self.board[np.where(self.board < 0)].size == 0 or self.board[np.where(self.board > 0)].size == 0:
            return 1
        if self.gamesWithoutKillOrKing == 50:  # ( draw )
            return 1

        # TODO: same board postions more than 3 times ( draw )
        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        if self.board[np.where(self.board < 0)].size == 0:
            return (-1, -1, 1)
        elif self.board[np.where(self.board > 0)].size == 0:
            return (1, -1, 1)

        return (0, 0, 0)

    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])

    def takeAction(self, action, from_action = None):

        # remove opposite if killing

        # move from one pos to other

        if type(action) in [np.int64, int]:
            for obj in self.allowedActionsFull:
                if obj['to'] == action and from_action != None and obj['from'] == from_action:
                    action = obj
                    break

        newBoard = np.array(self.board)
        newBoard[action['from']] = 0
        newBoard[action['to']] = self.playerTurn
        if action['kill'] != None:
            newBoard[action['kill']] = 0
        # newBoard[action] = self.playerTurn

        newState = GameState(newBoard, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done)

    def render(self, logger):
        for r in range(8):
            logger.info([self.pieces[str(x)] for x in self.board[8 * r: (8 * r + 8)]])
        logger.info('--------------')
