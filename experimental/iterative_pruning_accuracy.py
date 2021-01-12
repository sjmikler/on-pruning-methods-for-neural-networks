from collections import defaultdict

from tools.collect_logs import recursive_collect_logs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

sns.set()

name2path = {
    'LTR, step 30%': ('data/VGG19_IMP03_ticket', 775908, 770423),
    'LTR LR rewind, step 20%': ('data/',
                                810090, 665923, 572909),
    'LTR LR rewind, step 30%': ('data/VGG19_IMP03_LRrew', 718963),
}
name2result = defaultdict(list)

plt.figure(figsize=(8, 5), dpi=200)

for name, (path, *eid) in name2path.items():
    if path.endswith('.yaml'):
        logs = yaml.safe_load_all(open(path, 'r'))
    else:
        logs = recursive_collect_logs(path)

    for exp in logs:
        if not exp:
            continue

        acc = 'acc' if 'acc' in exp else 'ACC'
        if not any(str(x) in str(exp) for x in eid):
            continue

        name2result[name].append(
            (exp['pruning_config']['sparsity'], exp[acc]))

for name in name2result:
    r = name2result[name]
    results = defaultdict(list)
    for k, v in r:
        k = round(k, 4)
        results[k].append(v)
    results = {k: results[k] for k in sorted(results)}
    keys = list(results.keys())

    print(f"{name}: {[len(x) for x in results.values()]} runs")
    print(results)

    line = plt.plot(
        keys,
        [np.mean(x) for x in results.values()],
        label=name,
        linewidth=3,
        # alpha=0.8,
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
plt.xlabel('sparsity (%)')
plt.ylabel('accuracy')

plt.xticks([0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999],
           [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.997, 0.998, 0.999])
plt.yticks([0.84, 0.86, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95])
plt.ylim(0.85, 0.955)
plt.xscale('logit', one_half='')

xticks = [0.3, 0.75, 0.88, 0.938, 0.969, 0.984, 0.992, 0.996, 0.998, 0.999]
xnames = [round(x * 100, 1) for x in xticks]
plt.xticks(xticks, xnames)

plt.tight_layout()
plt.savefig('images/iterative_pruning_accuracy.png')
plt.show()
