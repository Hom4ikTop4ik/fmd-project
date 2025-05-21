def parse(filepath:str):
    file = open(filepath, "r")
    coords = []
    for line in file:
        cordPairStr = line.removesuffix("\n").split(" ")
        cordPair = []
        for numstr in cordPairStr:
            cordPair.append(int(numstr))
        coords.append(cordPair)
    return coords
