# Author: Dimitrios Damopoulos
# MIT license (see LICENCE.txt in the top-level folder)

import unittest 

import os
from tempfile import mkstemp 
from os.path import exists

import numpy as np
from numpy import random
import string
from random import choice as std_choice

from single_neuron import neuron as neuron

csv_files_to_check_n = 100
max_column_header_len = 30
max_features_n = 100
max_samples_n = 100000

class CsvParsingTestCase(unittest.TestCase):
    
    @classmethod
    def synthetic_csv(cls, fd_out, N, m, sep, name_delim='', comment_char='#'):
        """
        Creates a synthetic CSV character stream with column headers, values and 
        extra lines which are either empty or they are comments.

        Args:
            fd_out (output stream): Where to write the CSV. Typically, that is a
                file descriptor.
            N (int): The number of samples
            m (int): The number of features per sample
            sep (str): The character that separates the values in the CSV file
            name_delim (str): An optional character to add before and after 
                every header column name.
            comment_char (str): When the first non-whitespace character of a 
                line is the character `comment_char', then that line should be 
                treated as a comment.

        Returns:
            A list of the headers (without the `name_delim')
            A np.ndarray of the values in the CSV.
        """

        charset = string.ascii_letters + string.punctuation + ' \t'
        charset = charset.replace(sep, '')
        charset = charset.replace(comment_char, '')

        if len(name_delim) > 0:
            charset = charset.replace(name_delim, '')

        headers = []

        while len(headers) < m:
            header_len = random.randint(1, max_column_header_len + 1)
            header = ''.join(std_choice(charset) for _ in range(header_len))
            headers.append(header.strip())

        values = 2000 * (random.rand(N, m) - 0.5)
        
        val_line_idx = 0
        is_header_written = False
        while val_line_idx < N: 

            # insert some comments
            if random.rand() < 0.1:
                line = comment_char + \
                        ''.join(std_choice(charset) for _ in range(100))
            # insert some black lines 
            elif random.rand() < 0.1:
                line = ''
            elif random.rand() < 0.1:
                line = '  '
            elif not(is_header_written):
                line = sep.join([name_delim + header + name_delim 
                                        for header in headers])
                is_header_written = True
            else:
                line = sep.join([str(element)
                                        for element in values[val_line_idx]])
                val_line_idx += 1

            fd_out.write(line + '\n')

        return values, headers

    def test_parse_csv(self):
        
        candidate_characters = string.ascii_letters + string.punctuation

        candidate_characters = candidate_characters.replace('.', '')
        candidate_characters = candidate_characters.replace('-', '')

        for _ in range(0, csv_files_to_check_n):

            N = random.randint(1, max_samples_n + 1)
            m = random.randint(1, max_features_n + 1)

            N = 10
            m = 3

            n_sep = ';'
            v_sep = ';'
            c_sep = '#'

            while c_sep == n_sep or c_sep == v_sep or n_sep == v_sep:
                n_sep = std_choice(candidate_characters)
                v_sep = std_choice(candidate_characters)
            
            if random.rand() < 0.5:
                n_sep = ''

            _, csv_fn = mkstemp()

            try:
                csv_fd = open(csv_fn, 'w')
                V1, H1 = self.synthetic_csv(csv_fd, N, m, sep=v_sep, 
                                            name_delim=n_sep, comment_char=c_sep)
                csv_fd.close()
                V2, H2 = neuron.parse_csv(csv_fn, sep=v_sep, name_delim=n_sep)
            finally:
                os.remove(csv_fn)

            self.assertEqual(H1, H2)
            self.assertTrue((V1 == V2).all())
