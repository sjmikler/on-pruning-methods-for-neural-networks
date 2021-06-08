import pandas as pd
import yaml

logs = yaml.safe_load_all(open('../yaml_download/logs_2021-05-10_20-34-17.yaml', 'r'))
logs = list(logs)

# %%

logs = pd.DataFrame(logs)

logs['TIME'] = logs['TIME'].fillna(value='2001.01.01 00:00')
logs['RND_IDX'] = logs['RND_IDX'].fillna(value='0')

# logs = pd.DataFrame([row for _, row in logs.iterrows() if 'warm' not in row.Name])
# logs = pd.DataFrame([row for _, row in logs.iterrows() if 'WRN52-1' in row.Name])
# logs = pd.DataFrame([row for _, row in logs.iterrows() if 'IMP' in row.Name])
# logs = pd.DataFrame([row for _, row in logs.iterrows() if 'FT' not in row.Name])
# logs = pd.DataFrame([row for _, row in logs.iterrows() if isinstance(row.Desc, str)])


groups = {name: data for name, data in logs.groupby(['Name'])}
group_newest = {name: max(data['TIME']) for name, data in groups.items()}
group_order = sorted(group_newest, key=lambda x: group_newest[x])

for name in group_order:
    print(name)
    group = groups[name]
    group = group.sort_values('TIME')
    accs = list(groups[name]['ACC'])
    ids = list(groups[name]['RND_IDX'])
    accs = [f"{a:<9.7f}" for a in accs]
    ids = [f"{int(a):>9}" for a in ids]
    print('\t'.join(accs[::-1]))
    print('\t'.join(ids[::-1]))
    # input()
    print()
