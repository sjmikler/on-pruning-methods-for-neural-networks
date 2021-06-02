import pathlib
import json
import black
import isort


def format_str(src):
    src = black.format_str(src, mode=black.FileMode(line_length=88))
    src = isort.code(src, config=isort.Config(profile="black"))
    return src


line_length = 88

for nb_path in pathlib.Path('.').glob('**/*.ipynb'):
    nb = nb_path.open()
    nb = json.load(nb)
    dest = None

    source = []

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if cell['source'] and cell['source'][0].startswith('#default_exp'):
                dest = cell['source'][0][len('#default_exp'):].strip()

            if cell['source'] and cell['source'][0].startswith('#export'):
                source.extend(cell['source'][1:])
                source.append('\n# %%\n\n')
    source = ''.join(source)
    source = format_str(source)

    if source:
        source = f'"""Source created in notebook: {nb_path}"""\n\n' + source

        if dest:
            py_path = pathlib.Path(dest)
        else:
            py_path = nb_path.with_suffix('.py')
        print(f"Creating {py_path}")
        f = py_path.open(mode='w')
        py_path.write_text(source)
        f.close()
