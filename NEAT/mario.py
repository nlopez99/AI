#!/usr/bin/env python3
import retro
import neat
import numpy as np 
import cv2
import pickle

def eval_genomes(genomes, config):

    for id, genome in genomes:
        observation = env.reset()
        action = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape
        inx, iny = int(inx / 8), int(iny / 8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fit = 0
        fit_current = 0
        frame = 0
        counter = 0

        done = False
        while not done:
            
            env.render()
            frame += 1

            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))
            img_array = np.ndarray.flatten(observation)

            nn_output = net.activate(img_array)
            observation, reward, done, info = env.step(nn_output)

            if info['xscrollLo'] <= 40:
                fit_current = -1  
            else: fit_current += reward

            if fit_current > current_max_fit:
                current_max_fit = fit_current
                counter = 0
            else: counter +=1

            if done or counter == 250:
                done = True
                print(id, fit_current)

            genome.fitness = fit_current


# make the open ai environment and reset
env = retro.make('SuperMarioBros-Nes', 'Level1-1')
env.reset()

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

# with open('winner.pickle', 'wb') as output:
#     pickle.dump(winner, output, 1)
