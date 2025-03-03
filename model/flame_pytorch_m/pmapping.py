import torch


mapping_h2f = {
    37: 0,
    66: 1,
    27: 2,
    41: 3,
    58: 4,
    17: 5,
    39: 6,
    48: 7,
    6: 8,
    21: 9,
    8: 10,
    52: 11,
    22: 12,
    25: 13,
    43: 14,
    40: 15,
    24: 16,
    20: [25, 26],
    59: 24,
    3: 23,
    26: 22,
    56: 21,
    10: 20,
    28: 19,
    60: [17, 18],
    53: 36,
    55: 37,
    54: 38,
    36: 39,
    16: 40,
    63: 41,
    62: 42,
    64: 43,
    34: 44,
    13: 45,
    0: 46,
    65: 47,
    5: 27,
    29: 28,
    51: 29,
    30: 30,
    4: 31,
    46: 32,
    45: 33,
    35: 34,
    57: 35,
    44: 48,
    7: 60,
    31: 49,
    49: 50,
    15: 51,
    42: 52,
    32: 53,
    14: 64,
    33: 54,
    2: 55,
    1: 56,
    67: 57,
    47: 58,
    12: 59,
    9: 61,
    50: 62,
    38: 63,
    23: 65,
    18: 66,
    61: 67
}


def map_points(coords: torch.Tensor):
    converted = torch.zeros(68, 2)
    for key in mapping_h2f.keys():
        if type(mapping_h2f[key]) == list:
            for el in mapping_h2f[key]:
                converted[el] = coords[key]
        else:
            converted[mapping_h2f[key]] = coords[key]
    return converted