from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
import numpy as np # For image-matrix/vector operations
import cv2 # For image reduction
import neat
import pickle
import graphviz
import time
import pprint
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc

def preprocess_frame(frame):
	frame = np.where(frame==85,0,frame)
	frame = np.where(frame==174,0,frame)
	frame = np.where(frame==167,252,frame)
	frame = np.where(frame==99,0,frame)
	# frame = np.where(frame==61,2,frame)
	return frame

def config_verbal(population):
    p = population
    print("config: ", p.config)
    # pp.pprint(vars(p.config))
    for each in vars(p.config):
        print(each)
        try:
            pp.pprint(vars(vars(p.config)[each]))
        except:
            print(vars(p.config)[each])
            
    return 0

def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot

oned_image = []
np.set_printoptions(linewidth=200)
def eval_genomes(genomes, config):
    
    totalframes =0
    inx,iny,inc = env.observation_space.shape # inc = color

    inx = int(inx/8)
    iny = int(iny/8)
    # image reduction for faster processing
    FPS = 24
    seconds = 60
    # fourcc = VideoWriter_fourcc(*'mp4v')
    # video = VideoWriter('./noise.mp4', fourcc, float(FPS), (inx, iny), False)
    print(iny, inx)


    for genome_id, genome in genomes:
        
        ob = env.reset() # First image
        random_action = env.action_space.sample()


        # 20 Networks
        net = neat.nn.RecurrentNetwork.create(genome,config)
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0

        # if (frame>2):
        # 	done = True
        # 	genome.fitness = 0.0
        # else:
        # 	done = False

        done = False
        while not done:
            # vid = env.render(mode='rgb_array') # Optional
            env.render()  # watch mario learn!
            # print(vid)



            # img = Image.fromarray(vid, 'RGB')
            # img.save('my.png')
            # img.show()
  
            frame+=1
            totalframes+=1
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)


            ob = cv2.resize(ob,(inx,iny)) # Ob is the current frame
            # print(ob.shape)
            # ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) # Colors are not important for learning
            # print(ob.shape)
            # ob = np.reshape(ob,(iny,inx))
            # print(ob.shape)
            # ob = preprocess_frame(ob)
            # print(ob)
            # print("\n")

            # if totalframes < int(FPS*seconds):
            # 	# print(ob.shape)
            # 	video.write(ob)
            # if totalframes == int(FPS*seconds):
            # 	print("finish render")
            # 	video.release()

            oned_image = np.ndarray.flatten(ob)
            neuralnet_output = net.activate(oned_image) # Give an output for current frame from neural network
            # print(neuralnet_output)
            output = neuralnet_output.index(max(neuralnet_output))
            # output = int(sum(neuralnet_output)-1.0)
            # print(output)
            ob, rew, done, info = env.step(output) # Try given output from network in the game
            # print(info['life'])
            fitness_current += rew
            if fitness_current>current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter+=1
                # count the frames until it successful

            # Train mario for max 250 frames
            if done or counter == 200 or (info['life'] != 2):
                done = True 
                print(genome_id,fitness_current)
            
            genome.fitness = float(fitness_current)
            # time.sleep(0.2)

if __name__ == "__main__":
    pp = pprint.PrettyPrinter(4)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config-feedforward')

    # filename = '{0}{1}'.format('neat-checkpoint-', '49')

    filename = '{0}{1}'.format('pop300stag20/pop300-stag-20-', '56')

    try:
    	p = neat.Checkpointer().restore_checkpoint(filename)
    except:
    	p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # config_verbal(p)

    # Save the process after each 10 frames
    p.add_reporter(neat.Checkpointer(10,filename_prefix='server_training/pop300-'))

    winner = p.run(eval_genomes)
    # draw_net(config, winner, True)


    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

