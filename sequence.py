import gzip

import numpy as np
import pandas as pd


watson_crick = {'A': 'T',
                'T': 'A',
                'C': 'G',
                'G': 'C',
                'U': 'A',
                'N': 'N'}

watson_crick.update({k.lower(): v.lower()
                     for k, v in watson_crick.items()})


def read_fasta(f, as_df=False):
    if f.endswith('.gz'):
        fh = gzip.open(f)
        txt = fh.read().decode()
    else:
        fh = open(f, 'r')
        txt = fh.read()
    fh.close()
    records = parse_fasta(txt)
    if as_df:
        return pd.DataFrame(records, columns=('name', 'seq'))
    else:
        return records


def parse_fasta(txt):
    entries = []
    txt = '\n' + txt.strip()
    for raw in txt.split('\n>'):
        name = raw.split('\n')[0].strip()
        seq = ''.join(raw.split('\n')[1:]).replace(' ', '')
        if name:
            entries += [(name, seq)]
    return entries


def write_fasta(filename, list_or_records):
    if isinstance(list_or_records, pd.DataFrame) and list_or_records.shape[1] == 2:
        list_or_records = list_or_records.values
    list_or_records = list(list_or_records)
    with open(filename, 'w') as fh:
        fh.write(format_fasta(list_or_records))


def write_fake_fastq(filename, list_or_records):
    with open(filename, 'w') as fh:
        fh.write(format_fake_fastq(list_or_records))


def format_fake_fastq(list_or_records):
    """Generates a fake header for each read that is sufficient to fool bwa/NGmerge.
    """
    fake_header = '@M08044:78:000000000-L568G:1:{tile}:{x}:{y} 1:N:0:AAAAAAAA'
    if isinstance(next(iter(list_or_records)), str):
        records = list_to_records(list_or_records)
    else:
        records = list_or_records

    max_value = 1000
    lines = []
    for i, (_, seq) in enumerate(records):
        tile, rem = divmod(i, max_value**2)
        x, y = divmod(rem, max_value)
        lines.extend([fake_header.format(tile=tile, x=x, y=y), seq.upper(), '+', 'G' * len(seq)])
    return '\n'.join(lines)


def write_fastq(filename, names, sequences, quality_scores):
    with open(filename, 'w') as fh:
        fh.write(format_fastq(names, sequences, quality_scores))


def format_fastq(names, sequences, quality_scores):
    lines = []
    for name, seq, q_score in zip(names, sequences, quality_scores):
        lines.extend([name, seq, '+', q_score])
    return '\n'.join(lines)


def list_to_records(xs):
    n = len(xs)
    width = int(np.ceil(np.log10(n)))
    fmt = '{' + f':0{width}d' + '}'
    records = []
    for i, s in enumerate(xs):
        records += [(fmt.format(i), s)]
    return records


def format_fasta(list_or_records):
    if len(list_or_records) == 0:
        records = []
    elif isinstance(list_or_records[0], str):
        records = list_to_records(list_or_records)
    else:
        records = list_or_records
    
    lines = []
    for name, seq in records:
        lines.extend([f'>{name}', str(seq)])
    return '\n'.join(lines)


def reverse_complement(seq):
    return ''.join(watson_crick[x] for x in seq)[::-1]
