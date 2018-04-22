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
        self.action_size = len(self.gameState.labels_array)

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
        self.labels_array = [
            "5649", "5642",
            "5849" , "5851", "5840", "5844",
            "6051", "6053", "6042", "6046",
            "6253", "6255", "6244",

            "4940", "4942", "4935",
            "5142", "5144", "5133", "5137",
            "5344", "5346", "5335", "5339",
            "5546", "5537",

            "4033", "4026",
            "4233", "4224", "4235", "4228",
            "4435", "4437", "4426", "4430",
            "4637", "4639", "4628",

            "3324", "3326", "3319",
            "3526", "3528","3517", "3521",
            "3728", "3730", "3719", "3723",
            "3930", "3921",

            "2417", "2410",
            "2617", "2619", "2608", "2612",
            "2819", "2810", "2821", "2814",
            "3021", "3023", "3012",

            "1708", "1710", "1703",
            "1910", "1901", "1912", "1905",
            "2112", "2103", "2114", "2107",
            "2314", "2305",

            "0801", "1001", "1003",
            "1203","1205","1405","1407"
        ]
        self.flipped_labels_array = self._get_flipped_labels()
        self.gamesWithoutKillOrKing = 0
        # self.winners = []
        self.playerTurn = playerTurn
        self.binary = self._binary()
        self.id = self._convertStateToId()
        self.allowedActions = self._allowedActions()
        self.isEndGame = self._checkForEndGame()
        self.value = self._getValue()
        self.score = self._getScore()


    def _get_flipped_labels(self):
        labels = []
        for action in self.labels_array:


            to_pos = action[2:]
            from_pos = action[:2]

            # to_pos = to_pos+'0' if len(to_pos) != 2 else to_pos
            # from_pos = from_pos + '0' if len(from_pos) != 2 else from_pos
            labels.append(to_pos+from_pos)
        return labels

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
            if next_row > 7:
                return allowed
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
            if next_row < 0:
                return allowed
            steps_away = i % 8
            perpendiular_position = 8 * next_row + steps_away

            if perpendiular_position % 8 == 0:
                allowed = allowed + [perpendiular_position + (1 * steps)]
            elif perpendiular_position % 8 == 7:
                allowed = allowed + [perpendiular_position - (1 * steps)]
            else:
                allowed = allowed + [perpendiular_position - (1 * steps),
                                     perpendiular_position + (1 * steps)]

        temp = []

        # print(self.playerTurn, i)
        for j in allowed:
            if j < 64 and steps > 1 and self.board[j] == 0:
                temp.append(j)
            elif j < 64 and steps == 1:
                temp.append(j)

        return temp

    def getKillActions(self, i, king=False):
        allowed = self.allowedMovesDiagonallyWithSteps(i, king=king)
        final_allowed = []
        kill_moves = []
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
                temp_allowed = [value for value in temp if value in temp_current]
                # if len(temp_allowed) == 0:
                #     allowed = allowed + temp_allowed
                if len(temp_allowed) != 0:
                    ind = str(i)
                    ind2 = str(temp_allowed[0])
                    ind = '0'+ind if len(ind) == 1 else ind
                    ind2 = '0'+ind2 if len(ind2) == 1 else ind2
                    kill_moves.append(ind+ind2)
                    # final_allowed = final_allowed + [{'from': i, 'to': temp_allowed[0], 'kill': j}]
                allowed_len -= 1
            elif self.board[j] == 0:
                ind = str(i)
                ind2 = str(j)
                ind = '0' + ind if len(ind) == 1 else ind
                ind2 = '0' + ind2 if len(ind2) == 1 else ind2
                final_allowed.append(ind+ind2)
                # final_allowed = final_allowed + [{'from': i, 'to': j, 'kill': None}]
        if len(kill_moves) == 0:
            return final_allowed
        else:
            print(kill_moves)
            return kill_moves

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
        for action in allowed:
            if self.playerTurn < 0:
                temp.append(self.labels_array.index(action))
            else:
                temp.append(self.flipped_labels_array.index(action))

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
        # print(self.playerTurn, self.allowedActions)
        if self.board[np.where(self.board < 0)].size == 0 or self.board[np.where(self.board > 0)].size == 0:
            return 1
        if self.gamesWithoutKillOrKing == 50:  # ( draw )
            return 1

        if len(self.allowedActions) == 0:
            return 1

        # TODO: same board postions more than 3 times ( draw )
        return 0

    def _getValue(self):
        # This is the value of the state for the current player
        # i.e. if the previous player played a winning move, you lose
        # if self.board[np.where(self.board < 0)].size == 0:
        #     return (-1, -1, 1)
        # elif self.board[np.where(self.board > 0)].size == 0:
        #     return (-1, -1, 1)

        if len(self.allowedActions) != 0:
            return (0,0,0)
        no_of_black_pieces = self.board[np.where(self.board > 0)].size
        no_of_white_pieces = self.board[np.where(self.board < 0)].size
        # print(self.playerTurn, no_of_black_pieces, no_of_white_pieces)
        if no_of_black_pieces == no_of_white_pieces:
            return (0,0,0)


        if self.playerTurn > 0:
            if no_of_black_pieces < no_of_white_pieces:
                return (-1, -1, 1)
            elif no_of_black_pieces > no_of_white_pieces:
                return (1, 1, -1)
        else:
            if no_of_black_pieces < no_of_white_pieces:
                return (1, 1, -1)
            elif no_of_black_pieces > no_of_white_pieces:
                return (-1, -1, 1)
        #
        # if len(self.allowedActions) == 0:
        #     return (-1, -1, 1)

        return (0, 0, 0)

    def _getScore(self):
        tmp = self.value
        return (tmp[1], tmp[2])

    def takeAction(self, action):

        # remove opposite if killing

        # move from one pos to other

        action = self.labels_array[action] if self.playerTurn < 0 else self.flipped_labels_array[action]
        from_pos, to_pos = self.get_from_to(action)
        from_pos, to_pos = int(from_pos), int(to_pos)

        newBoard = np.array(self.board)
        newBoard[from_pos] = 0
        newBoard[to_pos] = self.playerTurn
        if (from_pos+to_pos) % 2 == 0:
            newBoard[(from_pos+to_pos)//2] = 0
        # if action['kill'] != None:
        #     newBoard[action['kill']] = 0
        # newBoard[action] = self.playerTurn

        newState = GameState(newBoard, -self.playerTurn)

        value = 0
        done = 0

        if newState.isEndGame:
            value = newState.value[0]
            done = 1

        return (newState, value, done)

    def get_from_to(self, action):
        to_pos = action[2:]
        from_pos = action[:2]


        return from_pos, to_pos



    def render(self, logger):
        for r in range(8):
            logger.info([self.pieces[str(x)] for x in self.board[8 * r: (8 * r + 8)]])
        logger.info('--------------')
