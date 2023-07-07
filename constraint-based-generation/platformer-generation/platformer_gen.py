import cv2
import numpy as np
from numpy.random import randint, choice


class generator:
    
    def __init__(
        self,
        img_directory,
        tile_size=16,
        vertical=False,
        n=2    
    ):
        
        
        self.img_directory = img_directory
        self.tile_size = tile_size
        self.vertical = vertical
        self.n = n
        
    def propagate(self, random_start=False, size=100):
        tiles = self.load_and_split_image()
        
        hashed_tiles = [self.pHash(tile) for tile in tiles]

        unique_hashes = list(dict.fromkeys(hashed_tiles))


        hash_int_dict = dict([(unique_hashes[i], i) for i in range(len(unique_hashes))])
        hash_tile_dict = dict(zip(hashed_tiles, tiles))
        int_tile_dict = dict([(hash_int_dict[i], hash_tile_dict[i]) for i in unique_hashes])
        
        hash_to_int = [hash_int_dict.get(item,item)  for item in hashed_tiles]
        constraints = self.get_constraints(hash_to_int)
        
        final_level = []
        if random_start:
            starting_tile = randint(low=0, high=list(constraints.keys())[-1])
        else:
            starting_tile = 0
        final_level.append(starting_tile)
        current_tile = starting_tile
        for i in range(size):
            next_tile = choice(constraints[current_tile])
            final_level.append(next_tile)
            current_tile=next_tile
        
        level_to_tiles = cv2.hconcat([int_tile_dict[i] for i in final_level])    
        
        return(level_to_tiles)
        
        
    def load_and_split_image(self):
        level = cv2.imread(self.img_directory)
        height, width, _ = level.shape
        if not self.vertical:           
            M = height
            N = self.tile_size
        else:
            M = self.tile_size
            N = width
        tiles = [level[x:x+M,y:y+N,:] for x in range(0,level.shape[0],M) for y in range(0,level.shape[1],N)]
        
        return tiles
    

    def pHash(self, cv_image):
        imgg = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY);
        h=cv2.img_hash.pHash(imgg) # 8-byte hash
        pH=int.from_bytes(h.tobytes(), byteorder='big', signed=False)
        return pH
    
    def create_ngrams(self, number_list):
        gram = [(number_list*2)[i: i + self.n] for i in range(len(number_list))]
        
        return gram
    
    def get_constraints(self, number_list):
        ngram = self.create_ngrams(number_list)
        constraints = {}
        for pair in ngram:
            if pair[0] not in constraints:
                constraints[pair[0]] = [pair[1]]
            elif pair[1] not in constraints[pair[0]]:
                constraints[pair[0]].append(pair[1])
        return constraints