import argparse
import re
import json
from datetime import datetime


def get_text(nb, args):
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            if cell['source'][-1].startswith('fig.savefig'):
                m = re.search("fig.savefig\(\'(.+?)\'.*", cell['source'][-1])
                yield f'![](/assets/images/posts/{args.name}/{m.group(1)}#center)\n\n'

            if cell['source'][0].startswith('# exclude'):
                continue

            yield '```python\n'
            for s in cell['source']:
                if len(s.strip()) > 70:
                    print(f'Code too long: {s}')
                yield s
            yield '\n```\n\n'

            if cell['outputs']:
                assert len(cell['outputs']) == 1
                yield '<div class="output_block">\n<pre class="output">\n'
                try:
                    yield from cell['outputs'][0]['text']
                except:
                    print(f'Erroneous output cell: {cell["outputs"]}')
                yield '</pre>\n</div>\n\n'

        elif cell['cell_type'] == 'markdown':
            src = cell['source']
            if src[0].startswith('# '):
                yield '---\nlayout: post\ntitle: '
                yield src[0][1:].strip()
                yield '\ncomments: True\n---\n\n'
                src = src[1:]
            yield from src
            yield '\n\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Notebook to Blog Post Converter')
    parser.add_argument('-f', '--file', type=str)
    parser.add_argument('-n', '--name', type=str)
    args = parser.parse_args()
    assert args.file and args.name

    if not args.file.endswith('.ipynb'):
        args.file += '.ipynb'
    with open(args.file, 'r') as f:
        nb = json.load(f)

    dt = datetime.today().strftime('%Y-%m-%d')
    text = ''.join(get_text(nb, args))
    with open(f'{dt}-{args.name}.md', 'w') as f:
        f.write(text)
