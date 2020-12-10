from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

sns.set()

name2path = {
    # 'LTR with step size 30%': (
    #     'data/VGG19_IMP03_ticket/95871/log.yaml', 95871),
    'LTR with step size 30%': (
        'logs/VGG19_LTR30_20steps.yaml', 770423),
    # '2 LTR with step size 30%': (
    #     'logs/VGG19_lottery_rewinding.yaml', 22903, 637467, 835349, 285535),
    # 'LTR with step size 50%': (
    #     'temp/VGG19_looking_for_tickets/VGG19IMP05/log.yaml', 32631, 33072),
    # 'LTR with step size 90%': (
    #     'temp/VGG19_looking_for_tickets/VGG19IMP05/log.yaml', 12061, 24517),
}
name2result = defaultdict(list)

plt.figure(figsize=(7, 5), dpi=200)

truning_results = [(0.858239013184179, 0.9348999857902527),
                   (0.963589093088294, 0.9355000257492065),
                   (0.9793353475829005, 0.9332000017166138),
                   (0.9865153815421495, 0.9305999875068665),
                   (0.9906697463044347, 0.9294999837875366),
                   (0.9937944966040752, 0.9261999726295471),
                   (0.8580811526168598, 0.9375),
                   (0.9650592788653616, 0.9370999932289124),
                   (0.9800656212544946, 0.9355000257492065),
                   (0.9867722732720735, 0.9297999739646912),
                   (0.9906785856971634, 0.9297999739646912),
                   (0.993205453455853, 0.9239000082015991),
                   (0.9949888134238913, 0.9205999970436096),
                   (0.9965732121454255, 0.9122999906539917)]
plt.scatter(*zip(*truning_results), label='iterative truning')

for name, (path, *eid) in name2path.items():
    for x in yaml.safe_load_all(open(path, 'r')):
        if not x:
            continue

        if 'idx' in x:
            idx = 'idx'
        else:
            idx = 'IDX'

        if not x or all(str(e) not in x[idx] for e in eid):
            continue

        if 'acc' in x:
            name2result[name].append((x['pruning_config']['sparsity'], x['acc']))
        else:
            name2result[name].append((x['pruning_config']['sparsity'], x['ACC']))

for name in name2result:
    r = name2result[name]
    results = defaultdict(list)
    for k, v in r:
        results[k].append(v)
    results = {k: results[k] for k in sorted(results)}
    print(f"{name}: {[len(x) for x in results.values()]} runs")
    keys = list(results.keys())

    line = plt.plot(
        keys,
        [np.mean(x) for x in results.values()],
        label=name,
        linewidth=3,
        alpha=0.8,
    )
    plt.fill_between(
        keys,
        [min(x) for x in results.values()],
        [max(x) for x in results.values()],
        alpha=0.2,
        color=line[0].get_color(),
        linewidth=0,
    )

plt.legend(loc=3)
plt.xlabel('sparsity')
plt.ylabel('accuracy')
plt.ylim(0.85, 0.95)
plt.xscale('logit')
plt.xticks([0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999],
           [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999])
plt.xlim(0.4, 0.999)
plt.yticks([0.84, 0.86, 0.88, 0.9, 0.92, 0.93, 0.94, 0.96])

plt.tight_layout()
# plt.savefig('oneshot_pruning/plotting/plots/iterative_pruning_accuracy.png')
plt.show()
