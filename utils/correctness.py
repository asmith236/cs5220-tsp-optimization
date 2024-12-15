import argparse
import os

medium_path = [0, 35, 21, 99, 50, 8, 85, 10, 2, 36, 47, 89, 31, 28, 32, 98, 94, 30, 19, 56, 66, 23, 48, 24, 92, 76, 52, 40, 65, 14, 91, 22, 62, 26, 51, 39, 93, 96, 6, 72, 41, 25, 67, 29, 86, 59, 74, 75, 3, 95, 44, 83, 68, 42, 37, 5, 78, 11, 79, 43, 60, 97, 46, 77, 53, 69, 45, 82, 71, 16, 57, 12, 27, 54, 88, 63, 55, 1, 20, 18, 81, 84, 17, 61, 38, 73, 87, 80, 7, 49, 90, 4, 13, 9, 15, 34, 70, 33, 64, 58, 0]
large_path = [767, 143, 960, 52, 944, 581, 900, 27, 622, 902, 330, 885, 57, 168, 327, 102, 587, 235, 479, 908, 888, 872, 205, 856, 564, 86, 8, 671, 753, 411, 452, 927, 482, 745, 465, 495, 442, 393, 935, 836, 626, 708, 640, 501, 609, 190, 382, 276, 780, 976, 97, 304, 420, 202, 192, 204, 287, 82, 210, 978, 206, 83, 10, 922, 144, 284, 813, 397, 527, 481, 925, 981, 868, 788, 584, 70, 503, 96, 737, 857, 39, 880, 597, 838, 700, 104, 784, 726, 286, 847, 3, 964, 42, 18, 651, 860, 257, 542, 848, 861, 830, 942, 867, 403, 797, 24, 374, 928, 261, 329, 963, 244, 139, 437, 731, 520, 322, 576, 881, 441, 554, 949, 961, 391, 498, 703, 678, 256, 158, 620, 533, 443, 746, 172, 858, 705, 779, 247, 694, 865, 630, 693, 773, 932, 953, 152, 310, 223, 368, 242, 128, 551, 899, 44, 265, 460, 326, 323, 611, 490, 825, 712, 140, 791, 381, 35, 241, 786, 508, 717, 159, 500, 704, 250, 369, 413, 167, 680, 470, 535, 727, 653, 412, 852, 777, 335, 920, 946, 560, 188, 373, 658, 596, 80, 951, 599, 110, 410, 77, 819, 311, 131, 754, 365, 462, 135, 394, 59, 875, 592, 478, 84, 132, 855, 947, 829, 476, 467, 60, 212, 303, 991, 424, 849, 616, 869, 175, 748, 390, 328, 89, 639, 553, 809, 237, 318, 181, 804, 614, 367, 845, 741, 993, 950, 351, 973, 432, 217, 510, 81, 185, 147, 701, 916, 934, 781, 923, 979, 647, 334, 28, 291, 285, 936, 752, 695, 458, 483, 816, 198, 595, 165, 245, 325, 183, 733, 617, 107, 760, 422, 239, 969, 33, 30, 673, 504, 566, 222, 98, 720, 529, 569, 371, 719, 148, 32, 805, 635, 73, 711, 51, 641, 208, 173, 668, 594, 54, 451, 907, 224, 236, 525, 603, 385, 280, 350, 725, 409, 345, 793, 802, 905, 684, 229, 472, 965, 883, 718, 372, 240, 271, 445, 279, 294, 155, 85, 321, 735, 986, 734, 275, 841, 870, 36, 589, 112, 980, 681, 379, 401, 572, 962, 349, 715, 427, 540, 707, 13, 765, 644, 5, 583, 65, 590, 435, 952, 714, 194, 264, 815, 94, 621, 795, 480, 811, 672, 277, 959, 7, 176, 299, 447, 187, 238, 464, 179, 61, 69, 579, 787, 213, 669, 297, 117, 398, 806, 56, 473, 37, 538, 134, 506, 366, 877, 383, 919, 509, 166, 706, 339, 763, 186, 912, 526, 14, 958, 545, 751, 438, 347, 742, 736, 149, 578, 130, 46, 489, 456, 486, 975, 471, 794, 444, 783, 29, 943, 586, 826, 886, 41, 618, 182, 100, 808, 361, 101, 547, 375, 665, 555, 246, 150, 468, 513, 839, 687, 220, 574, 580, 331, 864, 126, 929, 454, 191, 610, 249, 127, 230, 170, 109, 831, 463, 306, 629, 895, 585, 782, 682, 926, 792, 531, 582, 201, 253, 995, 408, 446, 312, 218, 561, 174, 588, 514, 428, 516, 676, 137, 648, 738, 817, 657, 882, 197, 785, 823, 461, 853, 627, 414, 255, 307, 851, 103, 43, 911, 674, 433, 396, 631, 915, 111, 696, 633, 739, 904, 216, 66, 710, 488, 534, 757, 195, 801, 854, 341, 258, 491, 789, 21, 40, 556, 716, 642, 762, 873, 487, 9, 1, 23, 278, 475, 774, 105, 941, 221, 260, 803, 828, 698, 196, 262, 543, 113, 219, 301, 778, 348, 697, 211, 200, 675, 897, 305, 338, 193, 87, 415, 484, 636, 834, 58, 234, 337, 302, 917, 799, 231, 996, 570, 749, 459, 232, 724, 146, 948, 138, 423, 827, 600, 766, 956, 666, 360, 512, 355, 49, 283, 683, 835, 893, 115, 655, 78, 930, 434, 546, 933, 387, 562, 263, 161, 552, 281, 116, 628, 660, 846, 317, 392, 690, 643, 419, 764, 91, 125, 938, 378, 699, 308, 353, 768, 776, 38, 90, 537, 502, 214, 417, 332, 79, 859, 492, 945, 850, 184, 344, 796, 844, 602, 494, 615, 315, 119, 25, 573, 418, 814, 744, 606, 493, 983, 652, 209, 539, 359, 399, 68, 910, 840, 121, 358, 11, 496, 862, 686, 429, 274, 955, 266, 252, 67, 505, 156, 189, 384, 818, 289, 300, 598, 268, 790, 833, 436, 608, 517, 997, 730, 466, 565, 967, 440, 507, 994, 837, 541, 612, 6, 106, 22, 842, 747, 723, 939, 866, 233, 637, 670, 634, 896, 761, 71, 685, 550, 519, 295, 987, 548, 663, 254, 898, 740, 957, 439, 812, 63, 863, 114, 178, 370, 142, 4, 123, 972, 821, 662, 226, 469, 559, 17, 457, 497, 571, 288, 225, 455, 755, 76, 380, 874, 395, 363, 649, 992, 16, 377, 876, 646, 568, 485, 141, 352, 623, 713, 528, 203, 282, 759, 775, 450, 48, 593, 999, 267, 160, 346, 340, 729, 53, 691, 770, 376, 824, 93, 319, 659, 151, 99, 632, 431, 62, 511, 887, 357, 732, 426, 153, 990, 406, 954, 988, 154, 207, 575, 309, 292, 0, 722, 924, 118, 92, 702, 891, 227, 664, 750, 906, 977, 903, 820, 515, 605, 272, 832, 522, 901, 169, 316, 180, 692, 822, 124, 75, 2, 273, 524, 430, 982, 416, 645, 47, 807, 968, 228, 940, 145, 985, 88, 889, 591, 72, 136, 638, 892, 251, 607, 26, 74, 129, 215, 625, 890, 354, 15, 800, 879, 604, 400, 421, 518, 563, 499, 342, 971, 162, 624, 248, 619, 31, 558, 544, 931, 769, 259, 55, 362, 320, 133, 974, 474, 728, 878, 871, 199, 798, 50, 843, 405, 918, 688, 386, 407, 19, 921, 909, 324, 984, 122, 567, 425, 679, 404, 453, 970, 64, 677, 270, 108, 884, 549, 656, 743, 530, 34, 937, 343, 613, 523, 577, 772, 163, 536, 293, 913, 389, 650, 989, 661, 477, 157, 756, 532, 243, 356, 557, 689, 709, 298, 20, 364, 314, 45, 290, 171, 998, 177, 667, 601, 521, 95, 758, 336, 810, 448, 164, 313, 388, 269, 914, 894, 966, 449, 721, 296, 333, 120, 771, 402, 12, 654, 767]
small_path = [0, 21, 10, 8, 2, 17, 20, 18, 14, 22, 26, 6, 25, 7, 9, 15, 4, 13, 3, 16, 5, 11, 1, 24, 23, 19, 12]
small_path_dp_cuda = [0, 21, 10, 8, 2, 17, 20, 18, 14, 22, 6, 25, 7, 9, 15, 4, 13, 3, 16, 5, 11, 1, 24, 23, 19, 12]
tiny_path = [0, 4, 1, 5, 8, 2, 7, 6, 9, 3]

# need a sepearate test for dp cuda small because it cannot exceed n > 26 unlike the others at 27

# Define the correct outputs for datasets and algorithms
CORRECT_SOLUTIONS = {
     "tiny": {
        "brute": {"cost": 12.516978, "path": tiny_path},
        "dp": {"cost": 12.516978, "path": tiny_path},
        "greedy": {"cost": 12.516978, "path": tiny_path},
        "genetic": {"cost": 12.516978, "path": tiny_path},
        "dp_omp": {"cost": 12.516978, "path": tiny_path},
        "dp_numa": {"cost": 12.516978, "path": tiny_path},
        "greedy_cuda": {"cost": 12.516978, "path": tiny_path},
        "dp_cuda": {"cost": 12.516978, "path": tiny_path}
    },
    "small": {
        "brute": {"cost": 4.380456, "path": small_path},
        "dp": {"cost": 4.380456, "path": small_path},
        "greedy": {"cost": 4.380456, "path": small_path},
        "genetic": {"cost": 4.380456, "path": small_path},
        "dp_omp": {"cost": 4.380456, "path": small_path},
        "dp_numa": {"cost": 4.380456, "path": small_path},
        "greedy_cuda": {"cost": 4.380456, "path": small_path},
        "dp_cuda": {"cost": 4.365681, "path": small_path_dp_cuda}
    }
}

def parse_out_file(file_path):
    """Parse the .out file to extract the cost and path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract cost
    cost_line = next(line for line in lines if line.startswith("Cost:"))
    cost = float(cost_line.split(":")[1].strip())

    # Extract path
    path_line = next(line for line in lines if line.startswith("Path:"))
    path = [int(x) for x in path_line.split(":")[1].strip().split("->")]

    return cost, path

def is_correct(actual_cost, actual_path, expected_cost, expected_path):
    """Check if the cost and path are correct."""
    # Check cost
    if round(actual_cost, 2) > round(expected_cost, 2):
        return False

    # Check path (including rotations and reversals)
    n = len(expected_path)
    for i in range(n):
        rotated_path = expected_path[i:] + expected_path[:i]
        reversed_path = rotated_path[::-1]
        if actual_path == rotated_path or actual_path == reversed_path:
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Verify TSP algorithm correctness.")
    parser.add_argument("--alg", choices=["brute", "dp", "greedy", "genetic", "dp_omp", "dp_cuda", "dp_numa", "greedy_cuda"], required=True, help="Algorithm to verify.")
    parser.add_argument("--csv", choices=["tiny", "small", "medium", "large"], required=True, help="Dataset to use for verification.")

    args = parser.parse_args()

    # Get correct solution
    if args.csv not in CORRECT_SOLUTIONS or args.alg not in CORRECT_SOLUTIONS[args.csv]:
        raise ValueError(f"No correct solution found for dataset '{args.csv}' and algorithm '{args.alg}'.")

    correct_solution = CORRECT_SOLUTIONS[args.csv][args.alg]
    expected_cost = correct_solution["cost"]
    expected_path = correct_solution["path"]

    # Parse the .out file
    try:
        actual_cost, actual_path = parse_out_file(f"build/{args.alg}.out")
    except Exception as e:
        print(f"Error reading .out file: {e}")
        return

    # Verify correctness
    if not is_correct(actual_cost, actual_path, expected_cost, expected_path):
        print("Incorrect!")
        print(f"Expected: cost = {expected_cost}, path = {expected_path}")
        print(f"Got: cost = {actual_cost}, path = {actual_path}")

if __name__ == "__main__":
    main()
