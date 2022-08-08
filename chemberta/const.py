import torch
adj_max=80
fps_len=167
max_len=120

vocabulary = {'C': 1, 'c': 2, '1': 3, '(': 4, '-': 5, '2': 6, 's': 7, 'N': 8, '=': 9, ')': 10, 'n': 11, '[': 12,
                  '@': 13,
                  'H': 14, ']': 15, 'O': 16, 'S': 17, '3': 18, 'l': 19, 'B': 20, 'r': 21, '/': 22, '\\': 23, 'o': 24,
                  '4': 25,
                  '5': 26, '6': 27, '7': 28, '+': 29, '.': 30, 'I': 31, 'F': 32, '8': 33, '#': 34, 'P': 35, '9': 36,
                  'a': 37,
                  '%': 38, '0': 39, 'i': 40, 'e': 41, 'L': 42, 'K': 43, 't': 44, 'T': 45, 'A': 46, 'g': 47, 'Z': 48,
                  'M': 49,
                  'R': 50, 'p': 51, 'b': 52, 'X': 53}

known_drugs = ['O=C(NCCC(O)=O)C(C=C1)=CC=C1/N=N/C(C=C2C(O)=O)=CC=C2OCCOC3=CC=C(NC4=NC=C(C)C(NC5=CC=CC(S(NC(C)(C)C)(=O)=O)=C5)=N4)C=C3',
        'OCCOC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(NC(C)(C)C)(=O)=O)=C3)=N2)C=C1',
        'C1CCC(C1)C(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3',
        'CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N',
        'CCS(=O)(=O)N1CC(C1)(CC#N)N2C=C(C=N2)C3=C4C=CNC4=NC=N3',
        'C1CC1C(=O)NC2=NN3C(=N2)C=CC=C3C4=CC=C(C=C4)CN5CCS(=O)(=O)CC5',
        'CCC1CN(CC1C2=CN=C3N2C4=C(NC=C4)N=C3)C(=O)NCC(F)(F)F',
        'OC(COC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(NC(C)(C)C)(=O)=O)=C3)=N2)C=C1)=O',
        'O=C(NCCC(O)=O)C(C=C1)=CC=C1/N=N/C(C=C2C(O)=O)=CC=C2OCCOC3=CC=C(NC4=NC=C(C)C(NC5=CC=CC(S(N)(=O)=O)=C5)=N4)C=C3',
        'OC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(NC(C)(C)C)(=O)=O)=C3)=N2)C=C1',
        'OCCOC1=CC=C(NC2=NC=C(C)C(NC3=CC=CC(S(N)(=O)=O)=C3)=N2)C=C1',
        'CC1=CN=C(N=C1NC2=CC(=CC=C2)S(=O)(=O)NC(C)(C)C)NC3=CC=C(C=C3)OCCN4CCCC4',
        'C1CCN(C1)CCOC2=C3COCC=CCOCC4=CC(=CC=C4)C5=NC(=NC=C5)NC(=C3)C=C2']

cuda_available = torch.cuda.is_available()
print("If cuda available:", cuda_available)
cuda_num=4
dv = "cuda:" + str(cuda_num)
device = torch.device(dv)
torch.cuda.set_device(int(dv[-1]))
