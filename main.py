import retro
import numpy as np
import cv2 
import neat
import pickle
import pygame



# Activation function for the neural network
def toggle(x):
    return 0 if x <= 0 else 1


# Puts all genomes to the test. Returns no value.
def eval_genomes(genomes, config):
    # This loop goes through each genome in the population one at a time.
    for genome_id, genome in genomes:
        # Reset the emulator and take a screenshot (2-D list of rgb values).
        ob = env.reset()
        # Create the neural network model from our config file's specifications.
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        # Variables to be used per genome, in the 'game loop' below
        current_x, frame, counter, reward_bag, prev_frame_x = 0, 0, 0, 0, 0

        # Display Purposes only ---------------------------------------------
        # This adds a second window to visualize what the agent can see.
        # cv2.add('display', cv2.WINDOW_NORMAL)
        
        # This is the 'game loop'. Each iteration of this loop is 1 frame on the
        #  emulator.
        done = False
        while not done:
            # If the machine can handle it, lets run the emulator at 1000 FPS.
            clock.tick(1000)
            # Show a window with the emulator in it.
            env.render()
            frame += 1

            # Divide our full-resolution screenshot of the emulator screen by 8.
            ob = cv2.resize(ob,  None, fx=0.125, fy=0.125)
            # Convert the full-color image to grayscale. This means there is now only
            #  one number for color, rather than 3 (r, g, b).
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)

            # Crop screenshot to only show us the important portion of the screen
            ob = ob[8:-2,4:]

            # Map pixel greyscale values to be (-4, 4), instead of (0, 255).
            for i, px in enumerate(ob):
                percentage = px / 255
                ob[i] = (percentage * 8) - 4

            # Display Purposes only ---------------------------------------------
            # cv2.imshow('display', ob)
            # cv2.waitKey(1)

            # Turn the 2-Dimensional observation into a 1-D array.
            ob = np.ndarray.flatten(ob)

            # Put our input information through the neural network (fwd propogation).
            nnOutput = net.activate(ob)

            # Edit our output (3 output nodes) to be a part of a 12-index list.
            # Each of these 12 indeces corresponds to a button on the controller.
            # Mario is able to jump, sprint, and move to the right. That's it.
            nnOutput = [toggle(nnOutput[0]), 0, 0, 0, 0, 0, 0, toggle(nnOutput[1]), toggle(nnOutput[2]), 0, 0, 0]
            
            # nnOutput = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

            # Step the environment forward by one frame, with the AI's controller
            #  inputs taken into account.
            ob, rew, done, info = env.step(nnOutput)

            # Store the cumulative reward of this run. We will add it to our
            #  genome's fitness at the end of the run.
            reward_bag += rew

            # Read Mario's X position from the emulator
            current_x = info['xscrollLo']
            
            # Since the address at 'xscrollLo' in the emulator is not just a cumulative running
            #  count of Mario's x value, we have to check against the previous frame to
            #  see if Mario is making progress on the X axis...
            if not abs(current_x - prev_frame_x) > 1:
                # If mario is farther now than he was last frame...
                if current_x > prev_frame_x:
                    prev_frame_x = current_x
                    # 2 'points' per pixel traveled to the right
                    reward_bag += 2
                    # Reset his frame limit counter, since he is improving on his X-position.
                    counter = 0
                else:
                    # If he is not getting any further to the right, his frame limit counter
                    #  goes up.
                    counter += 1
            else:
                prev_frame_x = current_x

            # If Mario hasnt moved further to the right within the last 180 frames...
            if done or counter >= 180:
                # Conclude his turn, end the loop.
                done = True
                # This genome's fitness is our rewards from X value increase only.
                genome.fitness = reward_bag
                # print(f'Genome ID: {genome_id}\t\tFitness: {genome.fitness}')

            ##################################################################
            #                                                                #
            #         v   v   v     DEBUGGING STUFF      v   v   v           #
            #                                                                #
            ##################################################################

            # print(f'xscrollHi = {info["xscrollHi"]}\t\txscrollLo = {info["xscrollLo"]}')



if __name__ == '__main__':
    # A pygame clock that we will use to set FPS.
    clock = pygame.time.Clock()

    # Setup the retro gym environment (emulator).
    env = retro.make('SuperMarioBros-Nes', 'Level1-1')

    # Configure the NEAT-Python module.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward.cfg')

    # Add our custom activation function so we can access it in the cfg.
    config.genome_config.add_activation('toggle', toggle)

    # Setup the population based on our config file.
    p = neat.Population(config)

    # We can use this function to restore a neural net from a checkpoint file.
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9')
    p.add_reporter(neat.StdOutReporter(True))

    # Setup a stats reporter for generational info to be printed in the console.
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Setup a 'checkpointer' that will save an entire generation to a file that
    #  we can load it from later. It saves every 10th generation.
    p.add_reporter(neat.Checkpointer(10))

    # This NEAT-Python function returns the highest fitness genome. (Also this
    #  is where we call our main function with the loop, 'eval_genomes').
    winner = p.run(eval_genomes)

    # After it's all over, once we have a 'winner' (highest fitness genome), 
    #  save that genome's configuration to a pickle file.
    # with open('winner.pkl', 'wb') as output:
        # pickle.dump(winner, output, 1)
