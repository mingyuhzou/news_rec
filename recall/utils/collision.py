import collections

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indices = len(set(all_indices_str.tolist()))
    return tot_item == tot_indices


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = collections.defaultdict(list)
    for i, index in enumerate(all_indices_str):
        index2id[index].append(i)

    collision_item_groups = []

    for i, g in index2id.items():
        if len(g) > 1:
            collision_item_groups.append(g)
    return collision_item_groups