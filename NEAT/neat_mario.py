#!/usr/bin/env python3
import gym_super_mario_bros
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np 
import multiprocessing as mp
import neat
import visualize
import pickle
import sys
import os
import cv2

# disable warnings
gym.logger.set_level(40)

class NEATMario:
    def __init__(self):
        # create number of generations and amount of parallel instances
        self.generations = 2000
        self.parallel_games = 2


    def fitness_func(self, genome, config, o):
        # create the environment
        game = gym_super_mario_bros.make('SuperMarioBros-v2')
        env = JoypadSpace(game, SIMPLE_MOVEMENT)
        try:
            # reset environment and create network from config file
            state = env.reset()
            neural_net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
            # frame count
            i = 0
            # starting mario position
            start_mario_distance = 40
            done = False

            # get shape of pixels
            inx, iny, inc = env.observation_space.shape
            inx, iny = int(inx / 8), int(iny / 8)

            while not done:
                # env.render() uncomment this to see mario play
                # resize image array and convert to grayscale
                state = cv2.resize(state, (inx, iny))
                state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
                state = np.reshape(state, (inx, iny))
                # flatten array so the network likes it
                state = state.flatten()

                # feed the state through the network and get max output
                output = neural_net.activate(state)
                action = output.index(max(output))

                # do the action from the net
                observation, reward, done, info = env.step(action)
                state = observation
                # increase frame count
                i += 1

                # check if 50 frames if mario moves and break from loop to restart if he hasn't
                if i % 50 == 0:
                    if start_mario_distance == info['x_pos']:
                        break
                    else: start_mario_distance = info['x_pos']

            # give a negative reward if mario didn't move else reward the distance he moved
            fitness = -1 if info['x_pos'] <= 40 else info['x_pos']

            # if at the end of the level dump the current genome to file
            if fitness >= 4000:
                pickle.dump(genome, open("winning_genome.pkl", "wb"))

            # put current fitness into queue
            o.put(fitness)
            env.close()

        except KeyboardInterrupt:
            env.close()
            sys.exit()


    def eval_genomes(self, genomes, config):
        # unpack genomes
        index, genomes = zip(*genomes)

        # create queue and process for amount of parallel games specified
        for i in range(0, len(genomes), self.parallel_games):
            output = mp.Queue()

            # create a process for each genome and pass it through the fitness function
            mario_processes = [mp.Process(target=self.fitness_func, args=(genome, config, output)) for genome in
                               genomes[i:i+self.parallel_games]]

            # start each process
            [process.start() for process in mario_processes]
            [process.join() for process in mario_processes]

            # get result from each process
            process_results = [output.get() for process in mario_processes]

            # set genome fitness to process result
            for n, r in enumerate(process_results):
                genomes[i + n].fitness = r


    def run(self, config_file):
        # boiler plate code for setting up neat with metrics refer to here: https://neat-python.readthedocs.io/en/latest/neat_overview.html
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                              neat.DefaultSpeciesSet, neat.DefaultStagnation, 
                              config_file)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(10))

        winner = p.run(self.eval_genomes, self.generations)
        # save genome that completed level
        pickle.dump(genome, open("winning_genome.pkl", "wb"))

        visualize.draw_net(config, winner, True)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)


    def main(self, config_file='mario_config_feedforward'):
        # get current directory of config file and use it
        current_dir = os.path.dirname(__file__)
        config_file_path = os.path.join(current_dir, config_file)
        self.run(config_file_path)


if __name__ == '__main__':
    mario = NEATMario()
    mario.main()