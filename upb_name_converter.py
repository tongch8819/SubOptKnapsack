import os


def convert():
    src_path = "./result/archive-24"
    task_list = ['adult']

    task_to_num_dict = {
        'adult': 100,
        'caltech': 100,
        'facebook': 1000,
        'youtube': 1000
    }

    for task in task_list:
        task_src_path = os.path.join(src_path, task, f"{task_to_num_dict[task]}")
        for seed in range(0, 20):
            instance_src_path = os.path.join(task_src_path, f"{seed}")
            for name in os.listdir(instance_src_path):
                algo, up, task, budget = name.strip()[:-5].split('-')
                up = up[:len(up) - 2]
                new_name = algo + '-' + up + '-' + task + '-' + budget + '.pckl'
                os.rename(os.path.join(instance_src_path,name), os.path.join(instance_src_path, new_name))

    pass


if __name__ == "__main__":
    convert()

