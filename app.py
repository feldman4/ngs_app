import os
import re
import sys
from glob import glob

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from natsort import natsorted
from slugify import slugify

from .sequence import (add_design_matches, read_fastq, reverse_translate_max,
                       try_translate_dna)
from .utils import assert_unique, csv_frame, dataframe_to_csv_string

# non-standard library imports delayed so fire app executes quickly (e.g., for help)

sample_table = 'samples.csv'
design_table = 'designs.csv'
stats_table = 'stats.csv'

command_lists = {
    'assemble': 'commands/0_assemble.list',
    'match': 'commands/1_match.list',
    'stats': 'commands/2_stats.list',
    'plot': 'commands/3_plot.list',
    }

# path to PEAR executable
ngs_app = 'ngs_app'

# columns
DESIGN_NAME = 'design_name' # name of design within insert
SAMPLE = 'sample' # slugified identifier
COUNT = 'count' # read count
TOTAL_ASSEMBLED_READS = 'total_assembled_reads' # reads successfully assembled by PEAR
TOTAL_WITH_ADAPTERS = 'total_with_adapters' # reads matching adapters
INSERT_MATCH = 'insert_match' # nearest match amino acid insert from design table
INSERT_DISTANCE = 'insert_distance' # Levenshtein distance to matched amino acid insert
INSERT_EQUIDISTANT = 'insert_equidistant' # number of other designed inserts at this edit distance
INSERT = 'insert' # translated amino acid sequence between adapters
INSERT_DNA = 'insert_dna' # DNA sequence between adapters
INSERT_DNA_MATCH = 'insert_dna_match' # matched DNA insert from design table
INSERT_DNA_DISTANCE = 'insert_dna_distance' # Levenshtein distance to matched DNA insert
INSERT_OUT_OF_FRAME = 'insert_out_of_frame' # is the insert out of frame?
INSERT_HAS_STOP = 'insert_has_stop' # is there a stop codon in the insert?

# if including barcodes
BARCODE = 'barcode' # amino acid barcode in design table
INSERT_BARCODE = 'insert_barcode' # barcode resulting from peptide purification at given terminus
MATCH_BARCODE = 'match_barcode' # barcode of matched insert
MISMAPPED_BARCODE = 'mismapped_barcode' # true if the barcode matches exactly, but not the insert
INSERT_FROM_BARCODE = 'insert_from_barcode' # the expected insert found by matching the barcode exactly
SUBPOOL_FROM_BARCODE = 'subpool_from_barcode' # the first subpool containing the barcode

# optional
SUBPOOL = 'subpool' # column in designs.csv
DESIGN_NAME = 'design_name' # column in designs.csv


def setup(design_table=design_table, sample_table=sample_table, min_counts=2, 
          include_barcodes=None):
    """Set up directories, validate metadata, and generate command lists for job submission.

    The analysis is configured via the `design_table` and `sample_table` csv files. `design_table`
    must contain an "insert_DNA" column with expected DNA sequences between adapters, or an 
    "insert" column with expected amino acid sequences (but not both). If barcodes are being 
    analyzed, a "barcode" column with amino acid sequences is required.

    :param design_table: csv table with one row per ordered design
    :param sample_table: csv table with one row per sequencing sample
    :param min_counts: minimum number of NGS reads to include sequence in matched table
    :param include_barcodes: whether to include barcode analysis; default is to include if 
        "barcode" is design table
    """
    

    # make directories
    for d in ('assembled', 'commands', 'results'):
        if not os.path.isdir(d):
            os.makedirs(d)

    df_designs = load_design_table()
    if BARCODE in df_designs and include_barcodes is None:
        print(f'WARNING: including barcodes since "barcode" column is in {design_table}, '
              f'explicitly set --include_barcodes=False to exclude')
        include_barcodes = True
    if INSERT_DNA not in pd.read_csv(design_table):
        print('Insert DNA not provided, so DNA metrics will not be calculated.')

    df_samples = (pd.read_csv(sample_table)
     .pipe(validate_sample_table, include_barcodes=include_barcodes)
    )
    validate_design_table(df_designs, include_barcodes=include_barcodes)

    # write commands
    # 0_assemble
    (pd.Series(prepare_pear_commands(df_samples))
     .to_csv(command_lists['assemble'], header=None, index=None)
    )

    # 1_match
    flags = f'--min_counts={min_counts}'
    if include_barcodes:
        flags += ' --include_barcodes'
    arr = []
    expected_results = []
    for sample in df_samples[SAMPLE]:
        expected_results += [f'results/{sample}.matched.csv']
        arr += [f'{ngs_app} match {sample} {flags} > results/{sample}.matched.csv']
    (pd.Series(arr).to_csv(command_lists['match'], header=None, index=None))

    # 2_stats
    cmd = f'{ngs_app} stats {" ".join(expected_results)} > {stats_table}'
    pd.Series([cmd]).to_csv(command_lists['stats'], header=None, index=None)
    
    # 3_plot
    cmd = f'{ngs_app} plot {" ".join(expected_results)} --output=figures/'
    pd.Series([cmd]).to_csv(command_lists['plot'], header=None, index=None)

    print(f"""
Pipeline ready. Submit to digs or run directly with:
  sh {command_lists['assemble']}
  sh {command_lists['match']}
  sh {command_lists['stats']}
  sh {command_lists['plot']}
    """[1:].strip())


def validate_sample_table(df_samples, include_barcodes):
    df_samples = df_samples.copy()
    allowed = r'[^-a-zA-Z0-9_\.]+'
    df_samples[SAMPLE] = [slugify(x, regex_pattern=allowed, lowercase=False) 
                            for x in df_samples[SAMPLE]]
    assert_unique(df_samples[SAMPLE])
    if include_barcodes:
        assert 'barcode_terminus' in df_samples
        assert set(df_samples['barcode_terminus']) <= {'N', 'C'}

    return df_samples


def load_design_table():
    
    df_designs = pd.read_csv(design_table)

    if INSERT in df_designs:
        if INSERT_DNA in df_designs:
            msg = (f'Cannot provide both "{INSERT}" (aa) and '
                   f'"{INSERT_DNA}" (DNA) in designs.csv')
            raise SystemExit(msg)
        df_designs[INSERT_DNA] = df_designs[INSERT].apply(reverse_translate_max)
    if SUBPOOL not in df_designs:
        df_designs[SUBPOOL] = 'all'
    return df_designs


def validate_design_table(df_designs, include_barcodes):
    df_designs = df_designs.copy()
    if INSERT_DNA in df_designs:
        assert_unique(df_designs[INSERT_DNA])
        df_designs[INSERT] = df_designs[INSERT_DNA].apply(try_translate_dna)
    if include_barcodes is True:
        it = df_designs[[INSERT, 'barcode']].values
        for insert, barcode in it:
            # assumes the insert is in-frame
            assert barcode in insert

    return df_designs


def print(*args, file=sys.stderr, **kwargs):
    from builtins import print
    print(*args, file=file, **kwargs)


def only_one_file(search):
    """Returns a single matching file, or else raises an appropriate error.
    """
    result = glob(search)
    if len(result) > 1:
        raise ValueError(f'Pattern {search} matched multiple files: {result}')
    if len(result) == 0:
        raise ValueError(f'Pattern {search} did not match any files')
    return result[0]


def prepare_pear_commands(df_samples):
    arr = []
    for _, row in df_samples.iterrows():
        r1 = only_one_file(f'fastq/*{row["fastq_name"]}*_R1*fastq*')
        r2 = only_one_file(f'fastq/*{row["fastq_name"]}*_R2*fastq*')        
        output = f'assembled/{row["sample"]}'
        cmd = f'pear -f {r1} -r {r2} -o {output}'
        arr += [cmd]
    return arr
    f = f'{home}/submit/pear_commands.list'
    pd.Series(arr).to_csv(f, index=None, header=None)


def write_sample_table_from_drive(libraries=['L015']):
    """Test function.
    """
    from postdoc.drive import Drive
    drive = Drive()
    df_ngs_libraries = drive('NGS/libraries')
    (df_ngs_libraries
     .query('library == @libraries')
     .assign(fastq_name=lambda x: x['index_plate'] + '_' + x['index_well'])
     .assign(sample=lambda x: x['template'])
     .assign(adapter_5='CACCACAGCAGTGGCAGT')
     .assign(adapter_3='TAACTCGAGCACCACCAC')
     .assign(barcode_terminus='N')
     [[SAMPLE, 'fastq_name', 'adapter_5', 'adapter_3', 'barcode_terminus']]
     .to_csv(sample_table, index=None)
    )


def write_design_table_from_chip(include_barcodes=True):
    """Test function.
    """
    chip_table = '/home/dfeldman/flycodes/chip_orders/chip137_design.csv'
    sources = ['foldit_monomers', 'foldit_oligomers', 'DK_beta_barrels', 'TS_variants', 
    'CD98_binders', 'BH_IL6R_variants']
    cols = ['subpool', DESIGN_NAME, INSERT_DNA]
    if include_barcodes:
        cols += ['barcode']
    
    (pd.read_csv(chip_table)
    .query('source == @sources')
    .assign(insert_dna=get_chip_insert)
    .rename(columns={'source': 'subpool'})
    .assign(design_name=lambda x: x['cds_name'].str.split('_').str[0])
    [cols]
    .to_csv(design_table, index=None)
    )


def get_chip_insert(df):
    """Test function.
    """
    oligos = df['oligo'].str.upper()
    forward = df['forward_adapter'].str[:-3] # remove CGA so it's kept in insert 
    reverse = df['reverse_adapter']

    arr = []
    for a,b,c in zip(oligos, forward, reverse):
        arr += [a.split(b)[1].split(c)[0]]

    return arr


def annotate_inserts(df_inserts, df_designs, window=30, k=12):

    design_info = (df_designs
    .set_index(INSERT).drop(INSERT_DNA, axis=1)
    .rename(columns={BARCODE: MATCH_BARCODE})
    )

    cols = [SAMPLE, COUNT, TOTAL_WITH_ADAPTERS, TOTAL_ASSEMBLED_READS, 
            INSERT_DISTANCE, INSERT_EQUIDISTANT] 
    cols += list(design_info.columns)
    cols += [INSERT_MATCH, INSERT, INSERT_DNA, INSERT_DNA_MATCH, 
            INSERT_DNA_DISTANCE, INSERT_OUT_OF_FRAME, INSERT_HAS_STOP]


    return (df_inserts
    .pipe(add_design_matches, INSERT_DNA, df_designs[INSERT_DNA], window, k)
    .rename(columns={'design_match': INSERT_DNA_MATCH, 
                    'design_distance': INSERT_DNA_DISTANCE})
    .drop('design_equidistant', axis=1)
    # returns None when insert is out of frame
    .assign(**{INSERT: lambda x: x[INSERT_DNA].apply(try_translate_dna)})
    .assign(**{INSERT_OUT_OF_FRAME: lambda x: x[INSERT].isnull()})
    .pipe(add_design_matches, INSERT, df_designs[INSERT], window, k)
    .rename(columns={'design_match': INSERT_MATCH, 
                    'design_distance': INSERT_DISTANCE, 
                    'design_equidistant': INSERT_EQUIDISTANT})
    .join(design_info, on=INSERT_MATCH)
    .assign(**{INSERT_HAS_STOP: lambda x: x[INSERT].str.contains('\*').fillna(False)})
    [cols]
    )


def has_subpools():
    return SUBPOOL in pd.read_csv(design_table)


def annotate_match_barcodes(df_matches, df_designs, barcode_terminus):

    barcode_pat = {
        'N': '.?R([^RK]*K).*',
        'C': '.*K([^RK]*R).?',
    }

    design_barcode_to_insert = df_designs.set_index(BARCODE)[INSERT].to_dict()
    

    barcodes = list(df_designs['barcode'])
    mismapped_gate = (f'{INSERT_BARCODE} == @barcodes & ~{INSERT_HAS_STOP} '
                      f'& {INSERT_FROM_BARCODE} != {INSERT_MATCH}')

    df_matches[INSERT_BARCODE] = df_matches[INSERT].str.extract(barcode_pat[barcode_terminus])[0]
    df_matches[INSERT_FROM_BARCODE] = df_matches[INSERT_BARCODE].map(design_barcode_to_insert)
    df_matches[MISMAPPED_BARCODE] = df_matches.eval(mismapped_gate)

    # doesn't mean anything if there's just one subpool
    design_barcode_to_subpool = df_designs.set_index(BARCODE)[SUBPOOL].to_dict()
    df_matches[SUBPOOL_FROM_BARCODE] = df_matches[INSERT_BARCODE].map(design_barcode_to_subpool)

    return df_matches.fillna('')


def parse_inserts(reads, pat):
    inserts = []
    for x in reads:
        match = re.findall(pat, x)
        if len(match) == 1:
            inserts += match

    return (pd.Series(inserts)
    .value_counts().reset_index()
    .rename(columns={'index': INSERT_DNA, 0: COUNT})
    .assign(total_assembled_reads=len(reads))
    )


def match(sample, sample_table=sample_table, design_table=design_table, min_counts=2, 
          include_barcodes=False):
    """Match assembled DNA inserts to design library.
    """
    df_designs = (load_design_table()
     .pipe(validate_design_table, include_barcodes=include_barcodes)
    )

    row = (pd.read_csv(sample_table)
     .pipe(validate_sample_table, include_barcodes=include_barcodes)
     .query('sample == @sample'))
    if len(row) != 1:
        msg = f'ERROR: expected one match for sample {sample}, found {list(row["sample"])}'
        raise SystemExit(msg)

    row = row.iloc[0]

    assembled_fastq = f'assembled/{row["sample"]}.assembled.fastq'
    reads = read_fastq(assembled_fastq)
    if len(reads) == 0:
        print(f'No reads detected for sample {sample}! Aborting.')
        return
    total_assembled = len(reads)

    pat = f'{row["adapter_5"].upper()}([ACGT]*?){row["adapter_3"].upper()}'

    df_matches = parse_inserts(reads, pat)
    total_with_adapters = df_matches['count'].sum()
    df_matches[TOTAL_WITH_ADAPTERS] = total_with_adapters
    df_matches = (df_matches
    .query('count >= @min_counts')
    .assign(sample=row[SAMPLE])
    )

    num_reads = df_matches['count'].sum()
    num_unique = len(df_matches)
    msg = (f'Loaded sample {row["sample"]}, '
           f'mapping {num_unique:,} unique reads ({num_reads:,} with >= {min_counts} reads; '
           f'{total_with_adapters:,} reads with adapters; {total_assembled:,} reads assembled)')
    print(msg)
    
    df_matches = annotate_inserts(df_matches, df_designs)

    if include_barcodes:
        df_matches = annotate_match_barcodes(df_matches, df_designs, row['barcode_terminus'])
    
    return dataframe_to_csv_string(df_matches)


def stats(*matched_tables):
    """Calculate summary statistics from result of `match` command.

    Example:
        $NGS_APP stats results/*matched.csv > stats.csv

    :param matched_tables: filename or pattern for match result tables; be sure to quote wildcards
    """
    df_matches = load_matched_tables(*matched_tables)

    cutoffs = 1e-2, 1e-3, 1e-4, 1e-5
    arr = []
    for sample, df in df_matches.groupby(SAMPLE):
        num_assembled = df[TOTAL_ASSEMBLED_READS].iloc[0]
        f = f'assembled/{sample}.unassembled.forward.fastq'
        num_reads = num_assembled + len(read_fastq(f))
        num_with_adapters = df[TOTAL_WITH_ADAPTERS].iloc[0]
        num_over_min = df[COUNT].sum()
        num_in_frame = df.query('~insert_out_of_frame')[COUNT].sum()
        num_no_stop = df.query('~insert_has_stop & ~insert_out_of_frame')[COUNT].sum()
        num_exact = df.query('insert_distance == 0')[COUNT].sum()
        num_exact_dna = df.query('insert_dna_distance == 0')[COUNT].sum()
        d = lambda a, b: np.round(a / b, 4) if b > 0 else 0
        info = {
            SAMPLE: sample,
            'total_reads': num_reads,
            'fraction_assembled': d(num_assembled, num_reads),
            'fraction_with_adapters': d(num_with_adapters, num_assembled),
            'fraction_over_min_count': d(num_over_min, num_with_adapters),
            'fraction_in_frame': d(num_in_frame, num_over_min),
            'fraction_no_stop': d(num_no_stop, num_in_frame),
            'fraction_exact_mapped': d(num_exact, num_no_stop),
            'fraction_exact_dna_mapped': d(num_exact_dna, num_exact),
        }
        
        
        col_order = list(info.keys())
        for cutoff in cutoffs:
            filt = (df[COUNT] / num_no_stop > cutoff)
            if 'match_barcode' in df:
                num_barcodes = df[filt]['match_barcode'].drop_duplicates().shape[0]
                info[f'num_barcodes_over_{cutoff:.0e}'] = num_barcodes
            
            if DESIGN_NAME in df and 'match_barcode' in df:
                barcode_counts = df[filt]['design_name'].value_counts()
                info[f'num_designs_with_1_barcode_over_{cutoff:.0e}'] = len(barcode_counts)
                info[f'num_designs_with_3_barcodes_over_{cutoff:.0e}'] = sum([x >= 3 for x in barcode_counts])
        cutoff_cols = natsorted(set(info.keys()) - set(col_order))
        info = {k: info[k] for k in col_order + cutoff_cols}
        arr += [info]

    # format so it's easy to read
    df_stats = pd.DataFrame(arr).astype(str).T
    df_stats.index.name = 'index'
    return df_stats.pipe(dataframe_to_csv_string, index=True)


def load_matched_tables(*matched_tables):
    matched_tables = natsorted([f for x in matched_tables for f in glob(x)])
    if len(matched_tables) == 0:
        raise SystemExit('ERROR: must provide at least one matched.csv table')

    return csv_frame(matched_tables)


def plot(*matched_tables, output='figures/', filetype='png', fuzzy_distance=15):
    """Generate QC plots from result of `match` command.

    The design-barcode count and barcode purity plots require "design_name" and "match_barcode" 
    columns. The sample cross mapping plot requires "subpool" column. Data used in plotting 
    is saved to a .csv table.

    Example:
        $NGS_APP stats results/*matched.csv

    :param matched_tables: filename or pattern for match result tables; be sure to quote wildcards
    :param output: prefix of saved figures and figure data
    :param filetype: extension for saved plot (png, jpg, pdf, etc)
    """
    print('Loading data...')
    df_matches = load_matched_tables(*matched_tables)
    df_designs = load_design_table()
    multiple_subpools = len(set(df_designs[SUBPOOL])) > 1

    os.makedirs(os.path.dirname(output), exist_ok=True)

    if INSERT_BARCODE in df_matches:
        non_redundant_barcodes = (df_designs
            .drop_duplicates([SUBPOOL, BARCODE])
            .drop_duplicates(BARCODE, keep=False)
            [BARCODE].pipe(list)
        )

        redundant_barcodes = set(df_designs[BARCODE]) - set(non_redundant_barcodes)
        if redundant_barcodes:
            print(f'Detected {len(redundant_barcodes)} barcodes repeated across subpools, '
                'these will be removed for purity and crossmapping analysis.')

    with sns.plotting_context('notebook'):
        try:
            fg, df_plot = plot_abundance(df_matches, df_designs, mode='insert')
            f = f'{output}rank_abundance.{filetype}'
            fg.savefig(f)
            df_plot.to_csv(f'{output}rank_abundance.csv', index=None)
            print(f'Saved rank abundance plot to {f}')
        except:
            pass

        if INSERT_BARCODE in df_matches:
            fg, df_plot = plot_abundance(df_matches, df_designs, mode='barcode')
            f = f'{output}rank_abundance_barcodes.{filetype}'
            fg.savefig(f)
            df_plot.to_csv(f'{output}rank_abundance_barcodes.csv', index=None)
            print(f'Saved rank abundance plot of barcodes to {f}')

        fig, df_plot = plot_distance_distribution(df_matches)
        f = f'{output}distance_distribution.{filetype}'
        fig.savefig(f, bbox_inches='tight')
        df_plot.to_csv(f'{output}distance_distribution.csv', index=None)
        print(f'Saved edit distance distribution heatmap to {f}')

        fg, df_plot = plot_length_distribution(df_matches, df_designs)
        f = f'{output}insert_length.{filetype}'
        fg.savefig(f)
        df_plot.to_csv(f'{output}insert_length.csv', index=None)
        print(f'Saved insert length histogram to {f}')

        if multiple_subpools:
            f = f'{output}cross_mapping.{filetype}'
            fig, df_plot = plot_crossmapping(df_matches, df_designs, mode='insert')
            fig.savefig(f, bbox_inches='tight')
            df_plot.to_csv(f'{output}cross_mapping.csv', index=None)
            print(f'Saved cross mapping heatmap to {f}')

            f = f'{output}cross_mapping_fuzzy_{fuzzy_distance}.{filetype}'
            fig, df_plot = plot_crossmapping(df_matches, df_designs, mode='insert', 
                                             max_insert_distance=fuzzy_distance)
            fig.savefig(f, bbox_inches='tight')
            df_plot.to_csv(f'{output}cross_mapping_fuzzy_{fuzzy_distance}.csv', index=None)
            print(f'Saved fuzzy cross mapping heatmap (insert distance within {fuzzy_distance}) '
                  f'to {f}')

            if MATCH_BARCODE in df_matches:
                f = f'{output}cross_mapping_barcodes.{filetype}'
                fig, df_plot = plot_crossmapping(df_matches, df_designs, mode='barcode')
                fig.savefig(f, bbox_inches='tight')
                df_plot.to_csv(f'{output}cross_mapping_barcodes.csv', index=None)
                print(f'Saved barcode cross mapping heatmap to {f}')


        if DESIGN_NAME in df_matches and MATCH_BARCODE in df_matches:
            gate = f'{INSERT_DISTANCE} == 0 & {INSERT_HAS_STOP} == False'
            for (sample, subpool), df in df_matches.query(gate).groupby([SAMPLE, SUBPOOL]):
                fig, df_plot = plot_detection_cutoffs_barcode(df)
                f = f'{output}design_barcode_counts_{sample}-{subpool}.{filetype}'
                fig.savefig(f, bbox_inches='tight')
                df_plot.to_csv(f'{output}design_barcode_counts_{sample}-{subpool}.csv')
                print(f'Saved design-barcode count heatmap ({sample}-{subpool}) to {f}')

            fg, df_plot = plot_barcode_purity(df_matches, df_designs)
            f = f'{output}barcode_purity.{filetype}'
            fg.savefig(f)
            df_plot.to_csv(f'{output}barcode_purity.csv', index=None)
            print(f'Saved barcode purity histogram to {f}')


def plot_abundance(df_matches, df_designs, mode='insert'):
    """Plot of log abundance (y-axis) vs oligo rank (x-axis) for exact amino acid sequences.
    If the design table includes subpool labels, make one plot per sample colored by subpool. 
    Otherwise, combine all samples onto one plot.
    Mode is either "insert" or "barcode"
    """
    if mode == 'insert':
        key = INSERT
        subpool = SUBPOOL
    elif mode == 'barcode':
        key = INSERT_BARCODE
        subpool = SUBPOOL_FROM_BARCODE
    else:
        raise ValueError(f'mode must be "insert" or "barcode", not {mode}')
    
    design_counts = df_designs.groupby(SUBPOOL).size().to_dict()

    if mode == 'insert':
        # only exactly matched inserts
        df_matches = df_matches.query('insert_distance == 0')
    elif mode == 'barcode':
        # only barcodes in the design table
        df_matches = df_matches.query(f'{INSERT_FROM_BARCODE} == {INSERT_FROM_BARCODE}')

    df_plot = (df_matches
    .groupby([SAMPLE, subpool, key])[COUNT].sum().reset_index()
    .assign(rank=lambda x: x.groupby([SAMPLE, subpool])[COUNT].rank(method='first', ascending=False))
    .sort_values([SAMPLE, subpool, 'rank']).reset_index(drop=True)
    [[SAMPLE, subpool, COUNT, 'rank', key]]
    )

    if len(df_plot) == 0:
        raise SystemExit('Nothing to plot!!')

    def plot(data, label, color):
        ax = plt.gca()
        ax.plot(data['rank'], data[COUNT], color=color, label=label)
        x = design_counts[label]
        ax.plot([x, x], [1, 100], color='red', ls=':', label='# designs')
        
    hue_kw = subpool
    row_kw = SAMPLE

    fg = (df_plot
    .pipe(sns.FacetGrid, row=row_kw, hue=hue_kw, height=4, aspect=1.5)
    .map_dataframe(plot)
    .add_legend()
    )

    for ax in fg.axes.flat[:]:
        ax.set_ylabel('Number of reads (exact aa match)')


    ax.set_xscale('log')
    ax.set_yscale('log')
    y0, y1 = ax.get_ylim()
    ax.set_ylim([1, y1])

    ax.set_xlabel(f'Matched {mode} rank')

    return fg, df_plot


def plot_detection_cutoffs_barcode(df_matches):
    """Number of designs with N barcodes above abundance cutoffs for a single sample.
    Abundance cutoffs are relative to number of inserts without stop codons.
    """
    df_counts = (df_matches
    .groupby([SUBPOOL, DESIGN_NAME, INSERT, MATCH_BARCODE])[COUNT].sum().reset_index()
    )

    num_no_stop = df_matches[COUNT].sum()

    arr = []
    fraction_cutoffs = 1e-2, 1e-3, 1e-4, 1e-5
    for subpool, df in df_counts.groupby(SUBPOOL):
        for cutoff in fraction_cutoffs:
            count_cutoff = max(int(num_no_stop * cutoff), 1)
            (df
            .query('@count_cutoff <= count')
            .groupby(DESIGN_NAME).size().value_counts().sort_index().reset_index()
            .rename(columns={'index': 'num_barcodes', 0: 'num_designs'})
            .assign(cutoff=f'{cutoff:.0e} ({count_cutoff} reads)')
            .assign(count_cutoff=count_cutoff)
            .assign(subpool=subpool)
            .pipe(arr.append)
            )

    barcode_counts = np.arange(1, pd.concat(arr)['num_barcodes'].max() + 1)
    df_plot = (pd.concat(arr)
    .drop_duplicates(['num_barcodes', 'num_designs', 'count_cutoff'])
    .pivot_table(index=['subpool', 'cutoff'], columns='num_barcodes', values='num_designs', aggfunc='first')
    .pipe(lambda x: x.reindex(columns=np.arange(1, max(x.columns) + 1)))
    .fillna(0).astype(int)
    .iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    )

    figsize = np.array([0.8, 0.8]) * df_plot.shape[::-1] + [2, 1]
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(df_plot, xticklabels=True, annot=True, ax=ax, 
                cbar=False, fmt='d')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xlabel('Number of barcodes')
    ax.set_ylabel('Read cutoff')
    ax.set_title(f'Number of designs with >= N barcodes\n({num_no_stop:,} reads without stop codons)')
    fig.tight_layout()
    return fig, df_plot


def plot_barcode_purity(df_matches, df_designs):
    """Purity is defined relative to the barcode detected in the insert 
    (i.e., the barcode that will be pulled down for MS).
    """
    allowed_barcodes = (df_designs
        .drop_duplicates([SUBPOOL, BARCODE])
        .drop_duplicates(BARCODE, keep=False)
        [BARCODE].pipe(list)
    )
    
    df_plot = (df_matches
    # only barcodes in the design table
    .query(f'{INSERT_BARCODE} == @allowed_barcodes')
    .query(f'{INSERT_FROM_BARCODE} == {INSERT_FROM_BARCODE}')
    .pivot_table(index=[SAMPLE, INSERT_BARCODE], 
                columns=MISMAPPED_BARCODE, values='count', aggfunc='sum')
    .reindex(columns=[False, True]).fillna(0).astype(int)
    .rename(columns={False: 'right_insert', True: 'wrong_insert'})
    .assign(purity=lambda x: x.eval('right_insert / (right_insert + wrong_insert)'))
    .reset_index()
    )
    df_plot.columns.name = ''

    fg = (df_plot
    .pipe(sns.FacetGrid, hue=SAMPLE)
    .map(plt.hist, 'purity', alpha=0.3, bins=np.linspace(0, 1, 30))
    .add_legend()
    )

    ax = fg.axes.flat[0]
    ax.set_xlabel('Purity')
    ax.set_ylabel('Number of barcodes')

    return fg, df_plot


def plot_crossmapping(df_matches, df_designs, mode='insert', max_insert_distance=0):
    """If mode=insert, determine crossmapping from entire insert. If mode=barcode, use 
    only barcodes that are unique across subpools.
    """
    if mode == 'insert':
        key = SUBPOOL
        # only matched inserts within edit distance (but not -1)
        df_matches = df_matches.query(f'0 <= {INSERT_DISTANCE} <= @max_insert_distance')
    elif mode == 'barcode':
        key = SUBPOOL_FROM_BARCODE
        allowed_barcodes = (df_designs
         .drop_duplicates([SUBPOOL, BARCODE])
         .drop_duplicates(BARCODE, keep=False)
         [BARCODE].pipe(list)
        )
        df_matches = (df_matches
         # only barcodes in the design table
         .query(f'{INSERT_FROM_BARCODE} == {INSERT_FROM_BARCODE}')
         # only barcodes that are unique across subpools
         .query(f'{INSERT_BARCODE} == @allowed_barcodes')
        )
    else:
        raise ValueError(f'mode must be "insert" or "barcode", not {mode}')

    fig, ax = plt.subplots()

    df_plot = (df_matches
    .pivot_table(index=SAMPLE, columns=key, values=COUNT, aggfunc='sum')
    .fillna(0).astype(int)
    )

    sns.heatmap(df_plot, square=True, annot=True, fmt='d', 
                xticklabels=True, yticklabels=True, cbar=False, ax=ax)

    plt.xticks(rotation=30)
    plt.yticks(rotation=0)

    return fig, df_plot


def plot_distance_distribution(df_matches):
    threshold = 5
    df_plot = (df_matches
    .query('~insert_has_stop')
    .assign(**{SUBPOOL: lambda x: x[SUBPOOL].fillna('not matched')})
    .pivot_table(index=[SAMPLE, SUBPOOL], 
                 columns=INSERT_DISTANCE, 
                 values='count', aggfunc='sum')
    .fillna(0).astype(int).T
    )
    df_counts = (pd.concat([
        df_plot[df_plot.index <= threshold].T, 
        df_plot[df_plot.index > threshold].sum().rename(f'>{threshold}')], 
        axis=1).T
    .rename({-1: 'not matched'})
    .T
    )

    figsize = np.array([1.2, 0.4]) * df_counts.shape[::-1] + [0, 1]
    fig, ax = plt.subplots(figsize=figsize)
    df_counts.pipe(sns.heatmap, annot=True, fmt=',', ax=ax, cbar=False)
    fig.tight_layout()
    plt.yticks(rotation=0)

    return fig, df_plot


def plot_length_distribution(df_matches, df_designs, focus_window=50):
    
    xlabel = 'Insert DNA length'

    cols = [SAMPLE, xlabel, SUBPOOL]
    df_matches = df_matches.assign(subpool=lambda x: x[SUBPOOL].fillna('unmapped'))

    df_plot = (df_matches
    .assign(**{xlabel: lambda x: x[INSERT_DNA].str.len()})
    .groupby(cols)[COUNT].sum().reset_index()
    .assign(sample_mode=lambda x: 
        x.sort_values(COUNT, ascending=False)
         .groupby(SAMPLE)[xlabel].transform('first'))
    .assign(sample_index=lambda x: x[SAMPLE].astype('category').cat.codes)
    )

    df_designs = df_designs.copy()
    df_designs[xlabel] = df_designs[INSERT_DNA].str.len()
    design_counts = df_designs.groupby([SUBPOOL, xlabel]).size().rename('count').reset_index()
    design_counts['sample_index'] = 1 + df_plot['sample_index'].max()
    design_counts['sample_mode'] = design_counts.sort_values('count').iloc[-1]
    design_counts['focus'] = 'full'
    design_counts[SAMPLE] = 'design table'

    fg = (pd.concat([
        df_plot.assign(focus='full'),
        df_plot.loc[lambda x: (x[xlabel] - x['sample_mode']).abs() < focus_window].assign(focus='top'),
        design_counts,
        design_counts.assign(focus='top'),
    ])
    .pipe(sns.FacetGrid, aspect=2, row=SAMPLE, col='focus', col_order=['full', 'top'], 
        hue=SUBPOOL, sharex=False)
    .map(plt.bar, xlabel, COUNT, alpha=0.6)
    .add_legend()
    )

    fg.axes.flat[0].set_yscale('log')
    df_lim = pd.concat([df_plot, design_counts])
    for ax in fg.axes[:, 0]:
        ax.set_ylabel('Read count')
        ax.set_xlim([df_lim[xlabel].min(), df_lim[xlabel].max()])
        ax.set_ylim([1, df_lim['count'].max()])

    return fg, df_plot


if __name__ == '__main__':

    # order is preserved
    commands = ['setup', 'match', 'stats', 'plot']
    # if the command name is different from the function name
    named = {
        # 'search': search_app,
        }

    final = {}
    for k in commands:
        try:
            final[k] = named[k]
        except KeyError:
            final[k] = eval(k)

    try:
        fire.Fire(final)
    except BrokenPipeError:
        pass
    

