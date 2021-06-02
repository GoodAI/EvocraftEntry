"""
    Evocraft entry
    
    Make sure to run the server first!
    
    Example commandlines for various experiments:
    
    # Default evolutionary novelty search experiment
    perform_experiment.py 
    
    # Evolutionary novelty search + automated planner
    perform_experiment.py --composition --predict
    
    # Evolutionary novelty search + automated planner - prediction
    perform_experiment.py --composition

"""

import grpc

import minecraft_pb2_grpc
from minecraft_pb2 import *

import numpy as np

import time
import copy

import argparse
import os

parser = argparse.ArgumentParser(description="Perform experiments for the Evocraft submission")
parser.add_argument('--composition', help="Use predictive composition of modules", action="store_true")
parser.add_argument('--pistons', help="Include piston blocks", action="store_true")
parser.add_argument('--predict', help="Predict what a composed module will do before testing", action="store_true")
parser.add_argument('--epochs', type=int, default=4000, help="How many steps to run the search")
parser.add_argument('--output_dir', type=str, default='output', help="Output directory for saved results")
parser.add_argument('--module_size', type=int, default=3, help="Size of the modules to evolve. If this is too large, it will fall outside of the testing area.")
parser.add_argument('--wait_interval', type=float, default=0.6, help="How long to wait between circuit evaluations")
args = parser.parse_args()

try:
    os.mkdir(args.output_dir)
except:
    pass

channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)

block_list = [ WATER, QUARTZ_BLOCK, PISTON, SLIME, STICKY_PISTON, REDSTONE_BLOCK, SAND, LAVA, TNT, REDSTONE_TORCH, OBSERVER, AIR ]
orientation_list = [NORTH, SOUTH, EAST, WEST]

if args.pistons:
    redstone_circuit_blocks = [ REDSTONE_WIRE, REDSTONE_TORCH, UNPOWERED_REPEATER, COBBLESTONE, AIR, PISTON, STICKY_PISTON, REDSTONE_BLOCK ]
else:
    redstone_circuit_blocks = [ REDSTONE_WIRE, REDSTONE_TORCH, UNPOWERED_REPEATER, COBBLESTONE, AIR ]    

MODULE_SIZE = args.module_size
INTERVAL = args.wait_interval

# Readout volume
XMIN = -100
XMAX = 100
ZMIN = -100
ZMAX = 100

# Insertion, deletion, mutation chances
P_INSERT = 0.4
P_DELETE = 0.1
P_CHANGE = 0.1

# Evaluation grid. Probably don't change - if things are too spread out it might hit chunk limits. Also, change XMIN, XMAX, etc if you make this larger
# Vertical would work better for the future...
POP_X = 3
POP_Y = 3
POPSIZE = POP_X * POP_Y 
SPACING = 40

# Based on the number of wires, should be 2 * 2^N
SIG_LENGTH = 32

# Initialize the signal that is going to be used to evaluate the circuit
signals = np.zeros((1,4,SIG_LENGTH)).astype(np.int16)

for i in range(SIG_LENGTH//2):
    signals[0,0,i] = i%2
    signals[0,1,i] = (i//2)%2
    signals[0,2,i] = (i//4)%2
    signals[0,3,i] = (i//8)%2

# Also evaluate the time reversal
signals[:,:,SIG_LENGTH//2:] = signals[:,:,:SIG_LENGTH//2][:,:,::-1]

signals = signals.repeat(POPSIZE, axis=0)

""" Setup of where the circuits will be tested. These should be far enough apart that circuits don't collide """
positions = []

for i in range(POP_X):
    for j in range(POP_Y):
        positions.append(np.array([SPACING*i-SPACING * (POP_X-1)//2, 4, SPACING*j-SPACING * (POP_Y-1)//2]))

archive = []
archive_patterns = []
archive_raw = []
a_lens = []

""" 
    SubGrid and Module definitions
    
    If you want to load and re-run patterns from the run, you will need copies of these definitions
    since the output files are pickled archives of these classes.
    
    These do depend on a few globals being in scope:
        - P_INSERT, P_DELETE, P_CHANGE, MODULE_SIZE
        - redstone_circuit_blocks, orientation_list

"""


""" This is a single NxN block within a larger circuit """
class SubGrid():
    def __init__(self, pos, blocks):
        if blocks is None:
            self.blocks = self.randomGrid(10+np.random.randint(20))
        else:
            self.blocks = np.array(blocks)
        self.pos = pos
        
    def randomGrid(self, N):
        blocks = []
        pos = [np.array([0,0,0])]
        bid = redstone_circuit_blocks[np.random.randint(len(redstone_circuit_blocks))]
        oid = orientation_list[np.random.randint(len(orientation_list))]
        blocks.append(np.array([0, 0, 0, bid, oid]))    
        half = (MODULE_SIZE-1)//2
        
        for i in range(N):
            j = np.random.randint(len(pos))
            p = pos[j].copy()
            k = np.random.randint(5)
            if k==0:
                p[1] += 2*np.random.randint(2)-1
            elif k<3:
                p[0] += 2*np.random.randint(2)-1
            else:
                p[2] += 2*np.random.randint(2)-1

            flatpos = [sp[0] + 100*sp[1] + 10000*sp[2] for sp in pos]
            fp = p[0]+100*p[1]+10000*p[2]

            if not fp in flatpos:
                if p[0]>=-half and p[0]<=half and p[2]>=-half and p[2]<=half and p[1]>=0 and p[1]<=3:
                    pos.append(p)
                    bid = redstone_circuit_blocks[np.random.randint(len(redstone_circuit_blocks))]
                    oid = orientation_list[np.random.randint(len(orientation_list))]
                    blocks.append(np.array([p[0],p[1],p[2], bid, oid]))
        return np.array(blocks)
    
    def getLayout(self, offset, edges):
        pos = self.pos + offset
        blocks = []
        # Include the construction of the module frame around this subgrid
        #
        # - Three units of redstone wire from the center each face
        # - A repeater which reads out the wire state
        #
        # The third unit of redstone should overlap adjacent modules, and is where the torch goes to
        # query the circuit
        
        half = (MODULE_SIZE-1)//2
        x0 = pos[0] - half
        x1 = pos[0] + half
        z0 = pos[2] - half
        z1 = pos[2] + half
        
        for z in range(z0,z1+1):
            for x in range(x0,x1+1):
                blocks.append(Block(position=Point(x=x, y=3, z=z), type=SAND))
    
        for i in range(3):
            if edges[0]:
                blocks.append(Block(position=Point(x=x0-1-i, y=4, z=z0+half), type=REDSTONE_WIRE, orientation=EAST))
            if edges[1]:
                blocks.append(Block(position=Point(x=x1+1+i, y=4, z=z0+half), type=REDSTONE_WIRE, orientation=EAST))
            if edges[2]:
                blocks.append(Block(position=Point(x=x0+half, y=4, z=z0-1-i), type=REDSTONE_WIRE, orientation=NORTH))
            if edges[3]:
                blocks.append(Block(position=Point(x=x0+half, y=4, z=z1+1+i), type=REDSTONE_WIRE, orientation=NORTH))

        if edges[0]:
            blocks.append(Block(position=Point(x=x0-2, y=4, z=z0+half-1), type=UNPOWERED_REPEATER, orientation=SOUTH))
        if edges[1]:
            blocks.append(Block(position=Point(x=x1+2, y=4, z=z0+half-1), type=UNPOWERED_REPEATER, orientation=SOUTH))
        if edges[2]:
            blocks.append(Block(position=Point(x=x0+half-1, y=4, z=z0-2), type=UNPOWERED_REPEATER, orientation=EAST))
        if edges[3]:
            blocks.append(Block(position=Point(x=x0+half-1, y=4, z=z1+2), type=UNPOWERED_REPEATER, orientation=EAST))
            
        for i in range(self.blocks.shape[0]):
            blocks.append(Block(position=Point(x=pos[0] + self.blocks[i,0],
                                               y=pos[1] + self.blocks[i,1],
                                               z=pos[2] + self.blocks[i,2]),
                                type=self.blocks[i,3],
                                orientation=self.blocks[i,4]))
                
        return blocks
    
    def getIn(self, offset):
        pos = self.pos + offset
        
        half = (MODULE_SIZE-1)//2
        x0 = pos[0] - half
        x1 = pos[0] + half
        z0 = pos[2] - half
        z1 = pos[2] + half
        
        return np.array([[x0-3, 4, z0+half], [x1+3, 4, z0+half], [x0+half, 4, z0-3], [x0+half, 4, z1+3]])
    
    def getOut(self, offset):
        pos = self.pos + offset
        
        half = (MODULE_SIZE-1)//2
        x0 = pos[0] - half
        x1 = pos[0] + half
        z0 = pos[2] - half
        z1 = pos[2] + half
        
        return np.array([[x0-2, 4, z0+half-1], [x1+2, 4, z0+half-1], [x0+half-1, 4, z0-2], [x0+half-1, 4, z1+2]])

""" 

This is a collection of smaller blocks wired together into a full circuit
Includes circuit layout, testing, etc code

"""  
        
class Module(): # Structure is a list of SubGrids
    def __init__(self, structure = None, signature = None, IO = [0,1,2,3]):
        self.links = []
        self.IO = IO
        if structure is None:
            self.randomStructure(20+np.random.randint(30))
        else:
            self.structure = structure
        
        self.signature = signature
    
    def getFullLayout(self, pos):
        blocks = []
        
        i = 0
        for s in self.structure:
            edges = np.ones(4)#zeros(4)
            blocks = blocks + s.getLayout(pos, edges)
            i += 1
        
        return blocks
    
    def getSubmoduleIOs(self):
        pos = np.zeros(3)
        allouts = []
        mod_id = []
        mod_edge = []
        
        i = 0
        for s in self.structure:
            allouts.append(s.getOut(pos))
            for j in range(4):
                mod_id.append(i)
                mod_edge.append(j)
                
            i += 1
            
        mod_id = np.array(mod_id)
        mod_edge = np.array(mod_edge)
        
        allouts = np.concatenate(allouts, axis=0)
        allids = allouts[:,0] + 1000 * allouts[:,1] + 1000000 * allouts[:,2]
        
        rowdict = {}
        for i in range(allids.shape[0]):
            if allids[i] in rowdict:
                rowdict[allids[i]] += 1
            else:
                rowdict[allids[i]] = 1
        
        keep_pts = []

        for i in range(allids.shape[0]):
            if rowdict[allids[i]] == 1:
                keep_pts.append(np.array([mod_id[i], mod_edge[i]]))
        return np.array(keep_pts)
        
    def getValidInPoints(self, pos):       
        allpts = []
        allouts = []
        directions = []
        for s in self.structure:
            allpts.append(s.getIn(pos))
            allouts.append(s.getOut(pos))
            for i in range(4):
                directions.append(i)
        
        directions = np.array(directions)
        allouts = np.concatenate(allouts, axis=0)
        allpts = np.concatenate(allpts, axis=0)
        allids = allouts[:,0] + 1000 * allouts[:,1] + 1000000 * allouts[:,2]
        
        rowdict = {}
        for i in range(allids.shape[0]):
            if allids[i] in rowdict:
                rowdict[allids[i]] += 1
            else:
                rowdict[allids[i]] = 1
        
        keep_pts = []
        keep_dirs = []
        for i in range(allpts.shape[0]):
            if rowdict[allids[i]] == 1:
                keep_pts.append(allpts[i])
                keep_dirs.append(directions[i])
        return np.array(keep_pts), np.array(keep_dirs)
    
    def getValidOutPoints(self, pos):
        allpts = []
        for s in self.structure:
            allpts.append(s.getOut(pos))
        allpts = np.concatenate(allpts, axis=0)
        allids = allpts[:,0] + 1000 * allpts[:,1] + 1000000 * allpts[:,2]
        rowdict = {}
        for i in range(allids.shape[0]):
            if allids[i] in rowdict:
                rowdict[allids[i]] += 1
            else:
                rowdict[allids[i]] = 1
        
        keep_pts = []
        for i in range(allpts.shape[0]):
            if rowdict[allids[i]] == 1:
                keep_pts.append(allpts[i])
        
        return np.array(keep_pts)
    
    def setTorches(self, pos, status):
        pts, _ = self.getValidInPoints(pos)
        pts = pts[self.IO]
        
        atlas = [ AIR, REDSTONE_TORCH ]
        
        blocks = []
        
        for i in range(4):
            blocks.append(Block(position=Point(x=pts[i][0], y=4, z=pts[i][2]), type=atlas[status[i]], orientation=UP))

        return blocks        
    
    def measure(self, corner, pos, full_read, RES):
        pts = self.getValidOutPoints(pos)[self.IO]
        results = []
        
        for i in range(4):
            coord = (pts[i][0]-corner[0])*RES + (pts[i][2]-corner[2])
            sig = full_read.blocks[coord]
            results.append(1*(sig.type == POWERED_REPEATER))
            
        return np.array(results)
    
    def randomStructure(self, N):
        if np.random.randint(2) == 0:
            self.structure = [SubGrid(np.array([0,0,0]), blocks=None)]
        else:
            self.structure = [SubGrid(np.array([0,0,0]), blocks=None), 
                              SubGrid(np.array([(MODULE_SIZE-1)//2+5,0,0]), blocks=None)]
            self.links = [[0,1], [1,0]]
            self.IO = np.random.permutation(6)[:4]
    
    def mutate_subgrid(self, idx):
        blocks = self.structure[idx].blocks.copy()
        pos = [blocks[i, :3] for i in range(blocks.shape[0])]
        blocks = [blocks[i] for i in range(blocks.shape[0])]
    
        for k in range(4):        
            # Insert
            if np.random.rand()<P_INSERT:
                j = np.random.randint(len(pos))
                p = pos[j].copy()
                k = np.random.randint(5)
                if k==0:
                    p[1] += 2*np.random.randint(2)-1
                elif k<3:
                    p[0] += 2*np.random.randint(2)-1
                else:
                    p[2] += 2*np.random.randint(2)-1

                flatpos = [sp[0] + 100*sp[1] + 10000*sp[2] for sp in pos]
                fp = p[0]+100*p[1]+10000*p[2]

                if not fp in flatpos:
                    if p[0]>=-(MODULE_SIZE-1)//2 and p[0]<=(MODULE_SIZE+1)//2 and p[2]>=-(MODULE_SIZE-1)//2 and p[2]<=(MODULE_SIZE-1)//2 and p[1]>=0 and p[1]<=3:
                        pos.append(p)
                        bid = redstone_circuit_blocks[np.random.randint(len(redstone_circuit_blocks))]
                        oid = orientation_list[np.random.randint(len(orientation_list))]
                        blocks.append(np.array([p[0],p[1],p[2], bid, oid]))

            if np.random.rand()<P_DELETE and len(pos)>1:
                j = np.random.randint(len(pos))
                del pos[j]
                del blocks[j]

            if np.random.rand()<P_CHANGE:
                j = np.random.randint(len(pos))
                blocks[j][3] = redstone_circuit_blocks[np.random.randint(len(redstone_circuit_blocks))]
                blocks[j][4] = orientation_list[np.random.randint(len(orientation_list))]

        self.structure[idx].blocks = np.array(blocks)
        
    def getSize(self):
        L = 0
        for s in self.structure:
            L = L + s.blocks.shape[0]
        
        return L
    
    # Attempts to merge this module with another one, connecting self_wire on this module
    # to other_wire on the other one
    def attemptMerge(self, other, self_wire, other_wire, ioset): 
        # Get the location of 'other' based on where it connects        
        # These are explicit world coordinates rather than module grid points
        self_inpts, self_dirs = self.getValidInPoints(np.zeros(3))
        other_inpts, other_dirs = other.getValidInPoints(np.zeros(3))
        
        # World coordinate of the input to the self-circuit
        # We also need to know which direction this wire extends from
        self_attach = self_inpts[self.IO[self_wire]]        
        self_dir = self_dirs[self.IO[self_wire]]
        self_io = self.getSubmoduleIOs()[self.IO[self_wire]]
        
        other_attach = other_inpts[other.IO[other_wire]]        
        other_dir = other_dirs[other.IO[other_wire]]
        other_io = other.getSubmoduleIOs()[other.IO[other_wire]]        
        
        # Invalid IO combination
        if (self_dir//2 != other_dir//2) or ( (self_dir+1)%2 != other_dir%2 ):
            return False
        
        contact = (self_attach - other_attach).astype(np.int32)
        
        if self_dir == 0:
            contact[0] += 2
        elif self_dir == 1:
            contact[0] -= 2
        elif self_dir == 2:
            contact[2] += 2
        else: 
            contact[2] -= 2
        
        merged = Module(copy.deepcopy(self.structure))
        merged.links = copy.deepcopy(self.links)
        for l in other.links:
            merged.links.append([l[0]+len(self.structure), l[1]])
        
        merged.links.append([self_io[0], self_io[1]])
        merged.links.append([other_io[0]+len(self.structure), other_io[1]])
        
        tick = 0
        for m2 in other.structure:
            p2 = m2.pos+contact
            for m1 in self.structure:
                tick += 1
                
                p1 = m1.pos
                # Size limits
                if p1[0]<-16 or p1[0]>16 or p1[2]<-16 or p1[2]>16:
                    return False
                
                # Collision
                if np.sum(np.abs(p1-p2)) == 0:
                    return False
                
            # Size limits
            if p2[0]<-16 or p2[0]>16 or p2[2]<-16 or p2[2]>16:
                return False
            
            m3 = copy.deepcopy(m2)
            m3.pos += contact
            merged.structure.append(m3)
                        
        new_io = []
        
        merged_inpts, merged_dirs = merged.getValidInPoints(np.zeros(3))
        for j in range(4):
            for i in range(len(merged_inpts)):
                if ioset[j][0] == 0:
                    if np.sum(np.abs(merged_inpts[i]-self_inpts[ioset[j][1]]))<1e-4:
                        new_io.append(i)
                else:
                    if np.sum(np.abs(merged_inpts[i]-other_inpts[ioset[j][1]]-contact))<1e-4:
                        new_io.append(i)
        
        if len(new_io) != 4:
            return False
        
        merged.IO = new_io
        return merged

""" Run a test sequence on a population of circuits all at once """

def testSignals(population, positions, signals): # signals is N x 4 x L    
    out_signals = np.zeros((signals.shape[0], 4, SIG_LENGTH))
    
    for t in range(SIG_LENGTH):   
        blocks = []
        for i in range(len(positions)):
            blocks = blocks + population[i].setTorches(positions[i], signals[i,:,t])        
        client.spawnBlocks(Blocks(blocks=blocks))
        
        time.sleep(INTERVAL)
        
        corner1 = np.array([XMIN, 4, ZMIN])
        corner2 = np.array([XMAX, 4, ZMAX])
        full_read = client.readCube(Cube(min=Point(x=corner1[0], y=corner1[1], z=corner1[2]),
                                         max=Point(x=corner2[0], y=corner2[1], z=corner2[2])))
                
        for i in range(len(positions)):            
            out_signals[i,:,t] = population[i].measure(corner1, positions[i], full_read, corner2[0]-corner1[0]+1)
                        
    # Clear torches
    blocks = []
    for i in range(len(positions)):
        blocks = blocks + population[i].setTorches(positions[i], np.zeros(4).astype(np.int32))
    client.spawnBlocks(Blocks(blocks=blocks))
    
    return out_signals      

""" Turn an IO pattern into a string so we can hash it """

def signature(response):
    x = ""
    r = response.ravel()
    for s in r:
        if s:
            x = x + "1"
        else:
            x = x + "0"
    return x

# Takes a set of modules, connections, input/output wires, and signal values, and simulates what should happen
#
# 'Modules' are a list of indexes into the archive
# 'Edges' are a list of ((Module, Wire), (Module, Wire)) pairs of pairs
# 'io' are four pairs of (Module, Wire) which correspond to inputs 1, 2, 3, 4
# signal is a 4-vector of what to put on all elements of io
#
# No (Module, Wire) pair can show up more than once in edges+io combined

# !The graph should be something that could occur physically given the shape of the modules
# Probably this means generating modules, edges, io from spatial arrangements rather than generating arbitrarily
# and then trying to arrange in space

def simulateGraph(modules, edges, io, signal):
    wire_states = np.zeros((len(modules), 4))
    
    for i in range(4):
        wire_states[io[i,0], io[i,1]] = signal[i]
    
    converged = False
    
    steps = 0
    
    while not converged:
        steps += 1
        converged = True
        
        for i in range(4):
            if wire_states[io[i,0], io[i,1]] == 0 and signal[i] == 1:
                converged = False
                wire_states[io[i,0], io[i,1]] = signal[i]
        
        for i in range(len(modules)):
            in_state = 0
            for j in range(4):
                in_state += 2**j * wire_states[i,j]
            in_state = int(in_state)
            
            out_state = archive_raw[modules[i]][:, in_state]
            
            for j in range(4):
                if out_state[j] != wire_states[i,j]:
                    converged = False
                    
                wire_states[i, j] = out_state[j]
                
                # Propagate to adjacent modules
                for e in edges:
                    if e[0][0] == i and e[0][1] == j:
                        if out_state[j] != wire_states[e[1][0], e[1][1]]:
                            converged = False
                        wire_states[e[1][0], e[1][1]] = out_state[j]
                    elif e[1][0] == i and e[1][1] == j:
                        if out_state[j] != wire_states[e[0][0], e[0][1]]:
                            converged = False
                        wire_states[e[0][0], e[0][1]] = out_state[j]
        
        if steps > 10000 and not converged:
            return False
        
    output = np.zeros(4)
    for i in range(4):
        output[i] = wire_states[io[i,0], io[i,1]]
        
    return output

""" 
attemptComposition implements the 'automated designer' part of this submission.

It tries to compose a pair of circuits to make a new circuit with new function 
We test as long as it takes for physical validity, but either return a success
or failure when it comes to finding novel function.
"""

def attemptComposition():
    valid = False
    
    # Find a physically valid combination
    while valid is False:
        valid = True
        i = np.random.randint(len(archive))
        j = np.random.randint(len(archive))

        edge0 = np.random.randint(4)
        edge1 = np.random.randint(4)

        io = []
        for k in range(4):
            success = False

            while success == False:
                success = True

                proposal = np.concatenate([np.random.randint(2, size=(1,)), np.random.randint(4, size=(1,))], axis=0)

                if proposal[0]==0 and proposal[1] == edge0:
                    success = False

                if proposal[0]==1 and proposal[1] == edge1:
                    success = False

                for l in range(len(io)):
                    if np.sum(np.abs(proposal-io[l]))<1e-2:
                        success = False
            io.append(proposal)
        io = np.array(io)

        combined = copy.deepcopy(archive[i]).attemptMerge(copy.deepcopy(archive[j]), edge0, edge1, io)
        
        if combined is False:
            valid = False
    
        simulated = []

        for k in range(SIG_LENGTH):
            response = simulateGraph([i,j], [ [[0,edge0], [1,edge1]] ], io, signals[0,:,k])
            if response is False:
                valid = False
            simulated.append(response)
    
    simulated = np.array(simulated).transpose(1,0)
    sig = signature(np.array(simulated))

    match = np.where(sig == np.array(archive_patterns))[0]
    
    # Only check the prediction against the archive if we're using the predictive planner
    if args.predict:
        if len(match)!=0:
            csize = combined.getSize()
            osize = archive[match[0]].getSize()

            if csize >= 0.5 * osize: # Only accept lack of novelty if its significantly smaller
                return 1, None # Failure to find a novel proposal
    
    return 2, combined # Found a novel proposal
    
# Clear everything

def clearAll(RANGE = 500):
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-RANGE, y=4, z=-RANGE),
            max=Point(x=RANGE, y=6, z=RANGE)
        ),
        type=LAVA))
    time.sleep(0.05)    
    
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-RANGE, y=4, z=-RANGE),
            max=Point(x=RANGE, y=20, z=RANGE)
        ),
        type=AIR))

    # Clear everything

    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-RANGE, y=3, z=-RANGE),
            max=Point(x=RANGE, y=3, z=RANGE)
        ),
        type=CLAY))

""" Main Code Loop """

population = [Module() for i in range(POPSIZE)]

for epoch in range(args.epochs):
    # Wipe all entities
    clearAll(120)
    
    blocks = []
    for i in range(len(positions)):
        blocks = blocks + population[i].getFullLayout(positions[i])
    
    client.spawnBlocks(Blocks(blocks=blocks))
    
    results1 = testSignals(population, positions, signals)
    
    clearAll(120)
    
    blocks = []
    for i in range(len(positions)):
        blocks = blocks + population[i].getFullLayout(positions[i])
    
    client.spawnBlocks(Blocks(blocks=blocks))
    
    results2 = testSignals(population, positions, signals)
    
    for i in range(len(population)):
        s1 = signature(results1[i])
        s2 = signature(results2[i])
        
        if s1==s2 and np.sum(results1[i][:,::-1]!=results1[i])==0:
            j = np.where(s1 == np.array(archive_patterns))[0]
            population[i].signature = s1

            if len(j)==0:
                archive_patterns.append(s1)
                archive.append(copy.deepcopy(population[i]))
                archive_raw.append(results1[i])
            else:
                if archive[j[0]].getSize()>population[i].getSize():
                    archive[j[0]] = copy.deepcopy(population[i])
    
    population = []
    for i in range(len(positions)):
        proposal = None
        best_result = 0
        
        if len(archive)>2:
            for j in range(5): # Try to find a composition 20 times before defaulting to mutation
                if best_result != 2:
                    result, p = attemptComposition()
                    
                if result == 2:
                    proposal = p
                    best_result = result
        
        if best_result == 2:
            population.append(copy.deepcopy(proposal))
        else:
            if np.random.randint(2)==0:
                population.append(Module())
            else:
                j = np.random.randint(len(archive))
                p = copy.deepcopy(archive[j])
                p.mutate_subgrid(np.random.randint(len(p.structure)))
                if len(p.structure)>1:
                    if np.random.randint(4) == 0:
                        del p.structure[np.random.randint(len(p.structure))]
                        inputs, _ = p.getValidInPoints(np.zeros(3))
                        p.IO = np.random.permutation(len(inputs))[:4]
                        
                population.append(p)
        
    a_lens.append(len(archive))
    
    np.save("%s/archive.npy" % (args.output_dir), np.array(archive))
    np.save("%s/archive_raw.npy" % (args.output_dir), np.array(archive_raw))
    f = open("%s/pattern.txt" % (args.output_dir), "w")
    for p in archive_patterns:
        f.write(p+"\n")
    f.close()
    np.savetxt("%s/count.txt" % (args.output_dir), a_lens)
