#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dataclasses
from enum import Enum
import itertools
import json
import sys
import difflib
from io import BytesIO
from typing import Optional, List, Tuple, Iterable
import tokenize
import keyword
from rkr_gst import run
import chardet


class LexMode(Enum):
    EXACT_MODE = 'exact'
    TOKENIZE_MODE = 'tokenize'
    IDENTIFIER_STRIP_MODE = 'id-strip'

    def __str__(self):
        return self.value


class MatchMethod(Enum):
    LCS = 'lcs'
    GST = 'gst'
    RKR_GST = 'rkr-gst'


def initialize_status(tokens):
    return [[token, False] for token in tokens]


def compare_token(tokens1, tokens2, line_1_num, line_2_num, same_tokens):
    try:
        if tokens1[line_1_num + same_tokens] == tokens2[line_2_num + same_tokens]:
            if not(tokens1[line_1_num + same_tokens][1] and tokens2[line_2_num + same_tokens][1]):
                return True
    except:
        return False
    return False


def check_same_length_match(matches, line_1_num, line_2_num):
    for line_3_num, match in enumerate(matches):
        match_line_1_start = match[0]
        match_line_2_start = match[1]
        match_len = match[2]
        if match_line_1_start <= line_1_num <= (match_line_1_start + match_len - 1):
            return False
        if match_line_2_start <= line_2_num <= (match_line_2_start + match_len - 1):
            return False
    return True


@dataclasses.dataclass()
class PythonSource:
    """
    To keep a part of Python code

    Attributes:
        file_name Source file name
        file_index Index within filename
        raw_lexemes Lexemes
        fingerprint_lexemes Lexemes with depersonated strings and IDs
    """
    file_name: str
    file_index: Optional[int]
    total_lexemes: int
    fingerprint_lexemes: List[str]

    @property
    def id_repr(self) -> str:
        return\
            "%s[%02d]" % (
                self.file_name,
                self.file_index) if self.file_index is not None else self.file_name

    def borrowed_fraction_from(
            self,
            other: 'PythonSource',
            minimal_match_length: int,
            match_method: MatchMethod
    ) -> Optional[float]:
        """Tells, what fraction of current source was (if it was)
        likely borrowed from another one"""
        if self is other or self.id_repr == other.id_repr:
            return None
        elif self.fingerprint_lexemes == other.fingerprint_lexemes:
            return 1.0
        if match_method == MatchMethod.GST:
            tokens_sequence_1 = self.fingerprint_lexemes.copy()
            tokens_sequence_2 = other.fingerprint_lexemes.copy()
            total_score = 0
            tokens_1 = initialize_status(tokens_sequence_1)
            tokens_2 = initialize_status(tokens_sequence_2)
            if len(tokens_2) < len(tokens_1):
                tokens_1, tokens_2 = tokens_2, tokens_1

            max_min = True

            while max_min:
                max_match = minimal_match_length
                matches = []
                for line_1_num, [_, is_match_1] in enumerate(tokens_1):
                    if not is_match_1:
                        for line_2_num, [_, is_match_2] in enumerate(tokens_2):
                            if not is_match_2:
                                same_tokens = 0
                                while compare_token(tokens_1, tokens_2, line_1_num, line_2_num, same_tokens):
                                    same_tokens += 1
                                if same_tokens == max_match:
                                    if check_same_length_match(matches, line_1_num, line_2_num):
                                        matches.append([line_1_num, line_2_num, same_tokens])
                                elif same_tokens > max_match:
                                    max_match = same_tokens
                                    matches = [[line_1_num, line_2_num, same_tokens]]

                for match in matches:
                    for enumerate_length in range(match[2]):
                        tokens_1[match[0] + enumerate_length][1] = True
                        tokens_2[match[1] + enumerate_length][1] = True

                    total_score += match[2]
                if max_match <= minimal_match_length:
                    max_min = False
            return float(total_score / len(self.fingerprint_lexemes))
        elif match_method == MatchMethod.LCS:
            self_marker = '\uE001'
            other_marker = '\uE002'
            self_lexemes = self.fingerprint_lexemes.copy()
            other_lexemes = other.fingerprint_lexemes.copy()
            common_size = 0
            resultative = True
            while resultative:
                sm = difflib.SequenceMatcher(
                    None,
                    self_lexemes,
                    other_lexemes,
                    False
                )  # type: ignore

                resultative = False
                for b in sm.get_matching_blocks():
                    self_index, other_index, match_size = tuple(b)
                    if match_size >= minimal_match_length:
                        resultative = True
                        common_size += match_size
                        self_lexemes[self_index: self_index + match_size] = [self_marker] * match_size
                        other_lexemes[other_index: other_index + match_size] = [other_marker] * match_size
            return float(common_size / len(self.fingerprint_lexemes))
        elif match_method == MatchMethod.RKR_GST:
            tokens_sequence_1 = self.fingerprint_lexemes.copy()
            tokens_sequence_2 = other.fingerprint_lexemes.copy()
            res = run(' '.join(tokens_sequence_1), ' '.join(tokens_sequence_2), minimal_match_length)
            return min(res, 1)
        else:
            raise ValueError(
                "Match method %s not implemented for Python sources" % match_method.value)

    @staticmethod
    def _lex_python_source(
            source_code: str, lex_mode: LexMode) -> Tuple[List[str], int]:
        """Get sequence of lexemes (depersonated or raw) from python source"""

        fingerprint_lexemes = []

        tokens = list(tokenize.tokenize(
            BytesIO(source_code.encode('utf-8')).readline))

        for ttype, tvalue, tstart, tend, tline in tokens:
            if ttype in (
                    tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE,
                    tokenize.NL, tokenize.ENDMARKER, tokenize.ERRORTOKEN
            ):
                continue

            ts = tvalue.strip()

            if lex_mode == LexMode.TOKENIZE_MODE:
                fingerprint_lexemes.append(ts)
            elif lex_mode == LexMode.IDENTIFIER_STRIP_MODE:
                if ttype == tokenize.NAME:
                    if keyword.iskeyword(ts):
                        fingerprint_lexemes.append(ts)
                    else:
                        fingerprint_lexemes.append('&id')
                elif ttype == tokenize.STRING:
                    fingerprint_lexemes.append('&""')
                elif ttype == tokenize.NUMBER:
                    fingerprint_lexemes.append('&num')
                elif ttype == tokenize.COMMENT:
                    fingerprint_lexemes.append('&#')
                else:
                    fingerprint_lexemes.append(ts)
            else:
                raise ValueError(
                    "Lex mode %s not implemented for Python sources" %
                    (lex_mode.value))

        return fingerprint_lexemes, len(tokens)

    @staticmethod
    def read_pythons_from_file(
            filename: str, lex_mode: LexMode) -> Iterable['PythonSource']:
        def read_nasty_file() -> str:
            try:
                with open(filename, 'r', encoding='utf-8') as tf:
                    return tf.read()
            except UnicodeDecodeError as ue:
                print(
                    "Author did not master UTF-8: %s" %
                    (filename), file=sys.stderr)
                with open(filename, 'rb') as bf:
                    bts = bf.read()
                    ec = chardet.detect(bts)
                    print(" - and with confidence of %f used %s" %
                          (ec['confidence'], ec['encoding']), file=sys.stderr)
                    return bts.decode(ec['encoding'])

        def read_pythons_from_notebook() -> Iterable['PythonSource']:
            """Too lazy to look for Jupyter API"""
            try:
                with open(filename, 'r', encoding='utf-8') as ipynb:
                    nbc = json.load(ipynb)
                    cells: Iterable = nbc['cells']
                    for c, n in zip(cells, itertools.count()):
                        if c['cell_type'] == 'code':
                            src = '\n'.join(
                                l if not l.startswith('%') else '#<%> ' + l
                                for l in c['source']
                            )
                            fl, tt = PythonSource._lex_python_source(
                                src, lex_mode)
                            yield PythonSource(filename, n, tt, fl)
            except Exception as e:
                print(
                    "Error reading %s" %
                    (filename),
                    repr(e),
                    file=sys.stderr)

        if filename.endswith('.ipynb'):
            yield from read_pythons_from_notebook()

        else:
            try:
                src = read_nasty_file()
                fl, tt = PythonSource._lex_python_source(src, lex_mode)
                yield PythonSource(filename, None, tt, fl)
            except Exception as e:
                print(
                    "Error reading %s" %
                    (filename),
                    repr(e),
                    file=sys.stderr)


if __name__ == '__main__':
    print("This is not a script, just a module", file=sys.stderr)
    exit(-1)
