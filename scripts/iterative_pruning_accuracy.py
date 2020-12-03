from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

sns.set()

name2path = {
    'LTR with step size 30%': (
        'temp/VGG19_looking_for_tickets/VGG19IMP03/log.yaml', 95871),
    'LTR with step size 50%': (
        'temp/VGG19_looking_for_tickets/VGG19IMP05/log.yaml', 32631, 33072),
    'LTR with step size 90%': (
        'temp/VGG19_looking_for_tickets/VGG19IMP05/log.yaml', 12061, 24517),
}
name2result = defaultdict(list)
sparsities = []

plt.figure(figsize=(7, 5), dpi=200)

for name, (path, *eid) in name2path.items():
    for x in yaml.safe_load_all(open(path, 'r')):
        if 'idx' in x:
            idx = 'idx'
        else:
            idx = 'IDX'

        if not x or all(str(e) not in x[idx] for e in eid):
            continue
        name2result[name].append((x['pruning_config']['sparsity'], x['acc']))
        sparsities.append(round(1 - x['pruning_config']['sparsity'], 4))

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
plt.xticks([0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995],
           [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995])
plt.xlim(0.4, 0.997)

plt.tight_layout()
# plt.savefig('oneshot_pruning/plotting/plots/iterative_pruning_accuracy.png')
plt.show()
