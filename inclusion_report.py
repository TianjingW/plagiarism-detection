#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import glob
import locale
import os
import shutil
import sys
import argparse
import multiprocessing
import multiprocessing.dummy
import subprocess
import time
from collections import Counter
from typing import List, Iterable, Tuple, Optional

import tzlocal
import tqdm

from sourcedrop import LexMode, PythonSource, MatchMethod
import report_export
from whoosh.analysis import KeywordAnalyzer
from whoosh.filedb.filestore import FileStorage
from whoosh.fields import *
from whoosh.qparser import QueryParser


def get_file_mdate(file_name: str) -> datetime.datetime:
    try:
        abs_file_name = os.path.abspath(file_name)
        output: subprocess.Popen = subprocess.Popen(
            ['git', 'log', '--date=iso', '--format="%ad"', '--', abs_file_name],
            stdout=subprocess.PIPE,
            cwd=os.path.split(abs_file_name)[0]
        )
        output.wait(5.0)

        if output.returncode != 0:
            raise UserWarning("Git return code was %d" % (output.returncode))

        bstdout: bytes = output.communicate()[0]
        stdout: Iterable[str] = bstdout.decode(
            locale.getpreferredencoding(False)).split('\n')
        stdout = [l.replace('"', '').replace("'", '').strip() for l in stdout]
        stdout = [l for l in stdout if len(l)]

        last_date: str = stdout[0]
        if last_date.endswith('00'):
            last_date = last_date[:-2] + ':00'

        return datetime.datetime.fromisoformat(last_date)
    except Exception as e:
        print("Error <<%s>> when getting git times for %s" %
              (str(e), file_name), file=sys.stderr)
        return datetime.datetime.utcfromtimestamp(os.path.getmtime(file_name))


def globs(path: str, min_date: Optional[datetime.datetime]) -> List[str]:
    filenames = []
    filenames.extend(
        glob.glob(
            os.path.join(
                path,
                '**',
                '*.py'),
            recursive=True))
    filenames.extend(
        glob.glob(
            os.path.join(
                path,
                '**',
                '*.ipynb'),
            recursive=True))
    filenames.sort()
    for filename in filenames:
        output: subprocess.Popen = subprocess.Popen(
            ['flake8', filename], stdout=subprocess.PIPE
        )
        output.wait(20.0)
        res = output.communicate()[0].decode(locale.getpreferredencoding(False)).strip()
        if len(res) > 0:
            print(res)
            c = Counter()
            for error in res.split('\n'):
                arr = error.split()
                if len(arr) > 2:
                    c[arr[1][0]] += 1
            total = ' '.join(['{}:{}'.format(k, v) for k, v in c.items()])
            print(filename, total)
    if min_date:
        filenames = [fn for fn in filenames if get_file_mdate(fn) >= min_date]

    return filenames


def get_python_sources(
        filenames: Iterable[str], min_lexemes: int, check_method: LexMode
) -> Iterable[PythonSource]:
    for fn in filenames:
        for ps in PythonSource.read_pythons_from_file(fn, check_method):
            if ps.total_lexemes >= min_lexemes:
                yield ps


def get_args() -> argparse.Namespace:
    # Thanks to https://gist.github.com/monkut/e60eea811ef085a6540f
    def valid_date_type(arg_date_str):
        """custom argparse *date* type for user dates values given from the command line"""
        try:
            given_time = datetime.datetime.strptime(arg_date_str, "%Y-%m-%d")
            tz_time = tzlocal.get_localzone().localize(given_time)
            return tz_time
        except ValueError:
            msg = "Given Date ({0}) not valid! Expected format, YYYY-MM-DD!".format(arg_date_str)
            raise argparse.ArgumentTypeError(msg)

    apr = argparse.ArgumentParser()
    apr.add_argument(
        "-gg",
        "--good-guys",
        help="Sources of good guys, who code",
        required=True
    )
    apr.add_argument(
        "-bg",
        "--bad-guys",
        help="Sources of presumably bad guys, who can steal",
        required=True
    )
    apr.add_argument(
        "-bt",
        "--borrow-threshold",
        help="Max amount of borrowed code to remain good",
        type=float,
        default=0.25
    )
    apr.add_argument(
        "-cm",
        "--check-method",
        type=LexMode,
        help="Check all lexemes or only structure (keywords, etc.) ones",
        default=LexMode.TOKENIZE_MODE.value,
        choices=list(LexMode)
    )
    apr.add_argument(
        "-mam",
        "--match-method",
        type=MatchMethod,
        help="match method to compare code",
        default=MatchMethod.RKR_GST.value,
        choices=list(MatchMethod)
    )
    apr.add_argument(
        "-ml",
        "--min-length",
        help="Minimal number of tokens in source to take it in account",
        type=int,
        default=20
    )
    apr.add_argument(
        "-wt",
        "--whoosh-threshold",
        help="Threshold of whoosh query result to consider",
        type=float,
        default=0.6
    )
    apr.add_argument(
        "-mml",
        "--min-match-length",
        help="Minimal length of text fragment to take in account",
        type=int,
        default=1
    )
    apr.add_argument(
        "-mbfd",
        "--min-bad-file-date",
        help="Oldest source file of bad guys to consider, older ones will be ignored; format: YYYY-MM-DD",
        type=valid_date_type
    )
    apr.add_argument(
        "-rf",
        "--report-file",
        help="OpenDocument spreadsheet to save the report",
        type=str,
        required=False
    )
    apr.add_argument(
        "-nm",
        "--no-multiprocessing",
        help="No multiprocessing to debug it easily",
        action='store_true',
        required=False
    )
    return apr.parse_args()


def _is_same_guy(bad_filename: str, good_filename: str,
                 bad_root: str, good_root: str) -> bool:
    """If files belong to the same guy to skip the check"""

    bad_root = os.path.normpath(bad_root). \
        lstrip(os.path.sep).rstrip(os.path.sep)
    good_root = os.path.normpath(good_root). \
        lstrip(os.path.sep).rstrip(os.path.sep)

    bad_filename = os.path.normpath(bad_filename). \
        lstrip(os.path.sep).rstrip(os.path.sep)
    good_filename = os.path.normpath(good_filename). \
        lstrip(os.path.sep).rstrip(os.path.sep)

    bad_filename = bad_filename.replace(bad_root + os.path.sep, '')
    good_filename = good_filename.replace(good_root + os.path.sep, '')

    return bad_filename.split(os.path.sep)[0] == \
        good_filename.split(os.path.sep)[0]


_minimal_match_length: int = 0


def compare_srcs(
        settings_bad_good: Tuple[PythonSource, PythonSource, float, MatchMethod]
) -> Tuple[str, str, Optional[float], float]:
    global _minimal_match_length
    bad, good, search_score, match_method = settings_bad_good
    borrowed_fraction = bad.borrowed_fraction_from(
        good, _minimal_match_length, match_method)
    return bad.id_repr, good.id_repr, borrowed_fraction, search_score


def compare_srcs_initializer(
        minimal_match_length: int):
    global _minimal_match_length
    _minimal_match_length = minimal_match_length


def workflow():
    args = get_args()
    analyzer = KeywordAnalyzer()
    schema = Schema(
        title=TEXT(stored=True),
        content=TEXT(stored=True, analyzer=analyzer)
    )

    if os.path.exists('./whoosh_index'):
        shutil.rmtree('./whoosh_index')

    storage = FileStorage('./whoosh_index')

    os.mkdir('./whoosh_index')
    ix = storage.create_index(schema)

    writer = ix.writer()

    borrow_threshold: float = args.borrow_threshold  # type: ignore
    check_method: LexMode = args.check_method  # type: ignore
    minimal_match_length: int = args.min_match_length  # type: ignore

    gs = globs(args.good_guys, None)  # type: ignore
    bs = globs(args.bad_guys, args.min_bad_file_date)  # type: ignore

    ml = args.min_length
    wt = args.whoosh_threshold
    match_method = args.match_method

    print("Looking for them...")
    good_sources = list(get_python_sources(gs, ml, check_method))
    bad_sources = list(get_python_sources(bs, ml, check_method))
    good_sources_dict = {}
    for g in good_sources:
        writer.add_document(title=g.file_name, content=' '.join(g.fingerprint_lexemes))
        good_sources_dict[g.file_name] = g
    writer.commit()
    tasks: List[Tuple[PythonSource, PythonSource, float, MatchMethod]] = []

    total_comparisons: int = 0
    done_comparisons: int = 0

    print("Capturing them...")
    with ix.searcher() as searcher:
        for b in bad_sources:
            query = QueryParser('content', ix.schema).parse(' '.join(b.fingerprint_lexemes))
            results = searcher.search(query, limit=len(good_sources))
            for index, res in enumerate(results):
                g = good_sources_dict[res['title']]
                if index <= wt * len(good_sources) and (not _is_same_guy(b.file_name, g.file_name, args.bad_guys, args.good_guys)):
                    tasks.append((b, g, res.score, match_method))
                    total_comparisons += 1
    print("Inquiring them...")
    start = time.time()
    if args.no_multiprocessing:  # type: ignore
        pool = multiprocessing.dummy.Pool(
            1,
            initializer=compare_srcs_initializer,
            initargs=[minimal_match_length]
        )
    else:
        pool = multiprocessing.Pool(
            initializer=compare_srcs_initializer,
            initargs=[minimal_match_length]
        )

    results = pool.imap_unordered(compare_srcs, tasks)
    print('{} cost {} seconds'.format(match_method.value.upper(), time.time() - start))
    borrowing_facts: List[Tuple[str, str, float]] = []

    tty = sys.stdout.isatty()
    if tty:
        results = tqdm.tqdm(results, total=total_comparisons)

    for bfn, gfn, bo, score in results:
        done_comparisons += 1
        if bo is not None and bo >= borrow_threshold:
            borrowing_facts.append((bfn, gfn, bo))
            (results.write if tty else print)(
                "%02d%% of %s borrowed from %s, %.4f score query from whoosh" % (int(100.0 * bo), bfn, gfn, score)
            )
    if args.report_file:
        report_export.export_csv_report(args.report_file, borrowing_facts)


if __name__ == '__main__':
    workflow()
