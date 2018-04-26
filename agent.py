# %matplotlib inline

import numpy as np
import random

import MCTS as mc
from game import GameState
from loss import softmax_cross_entropy_with_logits

import config
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl
import matplotlib.animation as animation
from matplotlib import style



class UserCheckers():

	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size
		self.labels_array = [
			"5649", "5642",
			"5849", "5851", "5840", "5844",
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
			"3526", "3528", "3517", "3521",
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
			"1203", "1205", "1405", "1407"
		]

	def act(self, state, tau):

		f = open("./communicate/input.txt", 'r+')
		action = f.readline()
		print("waiting for input")
		while  action == "":
			action = f.readline()
			if action == "restart":
				return action

		f.close()

		f = open("./communicate/input.txt", 'w')
		f.write("")
		f.close()

		pi = np.zeros(self.action_size)
		ind = self.labels_array.index(action)
		pi[ind] = 1
		value = None
		NN_value = None
		return (ind, pi, value, NN_value)


class User():

	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state, tau):
		f = open("./communicate/input.txt", 'r+')
		action = f.readline()
		print("waiting for input")
		while  action == "":
			action = f.readline()
			if action == "restart":
				return action
		action = int(action)
		f.close()

		f = open("./communicate/input.txt", 'w')
		f.write("")
		f.close()

		pi = np.zeros(self.action_size)
		pi[action] = 1
		value = None
		NN_value = None
		return (action, pi, value, NN_value)


		# return (ind, pi, value, NN_value)




class Agent():
	def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
		self.name = name

		self.state_size = state_size
		self.action_size = action_size

		self.cpuct = cpuct

		self.MCTSsimulations = mcts_simulations
		self.model = model

		self.mcts = None

		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []
		self.val_overall_loss = []
		self.val_value_loss = []
		self.val_policy_loss = []

	
	def simulate(self):

		lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
		self.mcts.root.state.render(lg.logger_mcts)
		lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

		##### MOVE THE LEAF NODE
		leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
		leaf.state.render(lg.logger_mcts)

		##### EVALUATE THE LEAF NODE
		value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

		##### BACKFILL THE VALUE THROUGH THE TREE
		self.mcts.backFill(leaf, value, breadcrumbs)


	def act(self, state, tau):

		if self.mcts == None or state.id not in self.mcts.tree:
			self.buildMCTS(state)
		else:
			self.changeRootMCTS(state)

		#### run the simulation
		for sim in range(self.MCTSsimulations):
			lg.logger_mcts.info('***************************')
			lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
			lg.logger_mcts.info('***************************')
			self.simulate()

		#### get action values
		pi, values = self.getAV(1)

		####pick the action
		action, value = self.chooseAction(pi, values, tau)

		nextState, _, _ = state.takeAction(action)

		NN_value = -self.get_preds(nextState)[0]

		lg.logger_mcts.info('ACTION VALUES...%s', pi)
		lg.logger_mcts.info('CHOSEN ACTION...%d', action)
		lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
		lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

		return (action, pi, value, NN_value)


	def get_preds(self, state):
		#predict_values the leaf
		inputToModel = np.array([self.model.convert_to_network_input(state)])

		preds = self.model.predict_values(inputToModel)
		value_array = preds[0]
		logits_array = preds[1]
		value = value_array[0]

		logits = logits_array[0]

		allowedActions = state.allowedActions
		# print(allowedActions, logits.shape)
		mask = np.ones(logits.shape,dtype=bool)
		mask[allowedActions] = False
		logits[mask] = -100

		#SOFTMAX
		odds = np.exp(logits)
		probs = odds / np.sum(odds) ###put this just before the for?

		return ((value, probs, allowedActions))


	def evaluateLeaf(self, leaf, value, done, breadcrumbs):

		lg.logger_mcts.info('------EVALUATING LEAF------')

		if done == 0:
	
			value, probs, allowedActions = self.get_preds(leaf.state)
			lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

			probs = probs[allowedActions]

			for idx, action in enumerate(allowedActions):
				newState, _, _ = leaf.state.takeAction(action)
				if newState.id not in self.mcts.tree:
					node = mc.Node(newState)
					self.mcts.addNode(node)
					lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
				else:
					node = self.mcts.tree[newState.id]
					lg.logger_mcts.info('existing node...%s...', node.id)

				newEdge = mc.Edge(leaf, node, probs[idx], action)
				leaf.edges.append((action, newEdge))
				
		else:
			lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

		return ((value, breadcrumbs))


		
	def getAV(self, tau):
		edges = self.mcts.root.edges
		pi = np.zeros(self.action_size, dtype=np.integer)
		values = np.zeros(self.action_size, dtype=np.float32)
		
		for action, edge in edges:
			pi[action] = pow(edge.stats['N'], 1/tau)
			values[action] = edge.stats['Q']

		pi = pi / (np.sum(pi) * 1.0)
		return pi, values

	def chooseAction(self, pi, values, tau):
		if tau == 0:
			actions = np.argwhere(pi == max(pi))
			action = random.choice(actions)[0]
		else:
			action_idx = np.random.multinomial(1, pi)
			action = np.where(action_idx==1)[0][0]

		value = values[action]

		return action, value

	def replay(self, ltmemory):
		lg.logger_mcts.info('******RETRAINING MODEL******')


		for i in range(config.TRAINING_LOOPS):
			minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

			training_states = np.array([self.model.convert_to_network_input(row['state']) for row in minibatch])
			training_targets = {'name_value': np.array([row['value'] for row in minibatch])
								, 'name_policy': np.array([row['AV'] for row in minibatch])}

			fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size = 32)
			lg.logger_mcts.info('NEW LOSS %s', fit.history)

			self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1],4))
			self.train_value_loss.append(round(fit.history['name_value_loss'][config.EPOCHS - 1],4))
			self.train_policy_loss.append(round(fit.history['name_policy_loss'][config.EPOCHS - 1],4))
		print("overal loss")
		f = open("./graphs/train_overall_loss.txt","w+")
		f.write(",".join([str(x) for x in self.train_overall_loss]))
		f.close()

		f = open("./graphs/value_head_loss.txt", "w+")
		f.write(",".join([str(x) for x in self.train_value_loss]))
		f.close()

		f = open("./graphs/policy_head_loss.txt", "w+")
		f.write(",".join([str(x) for x in self.train_policy_loss]))
		f.close()
		# plt.plot(self.train_overall_loss, 'k')
		# plt.plot(self.train_value_loss, 'k:')
		# plt.plot(self.train_policy_loss, 'k--')
        #
		# plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')


		display.clear_output(wait=True)
		display.display(pl.gcf())
		pl.gcf().clear()
		time.sleep(1.0)

		print('\n')
		# self.model.printWeightAverages()

	def predict(self, inputToModel):
		preds = self.model.predict_values(inputToModel)
		return preds

	def buildMCTS(self, state):
		lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
		self.root = mc.Node(state)
		self.mcts = mc.MCTS(self.root, self.cpuct)

	def changeRootMCTS(self, state):
		lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
		self.mcts.root = self.mcts.tree[state.id]