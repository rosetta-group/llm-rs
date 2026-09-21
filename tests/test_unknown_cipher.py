import unittest
import numpy as np
from experiments.codebook_free import artificial
from voynich.decipher import language_model
from voynich.unknown_cipher import representations, chunk_counts, solve
import random

class UnknownCipherTests(unittest.TestCase):
    def test_observed_representations(self):
        self.assertEqual(representations('ab cd'),[('characters',list('abcd')),('units',['ab','cd'])])
        self.assertEqual(representations('abc'),[('characters',list('abc'))])

    def test_variable_control_roundtrip(self):
        plain='questoeuntestodicontrollo'*10
        cipher,key=artificial(plain,random.Random(1))
        self.assertEqual(''.join(key[t] for t in cipher.split()),plain)
        self.assertEqual({len(c) for c in key.values()},{1,2})

    def test_public_only_interface_and_determinism(self):
        texts=['questo e un piccolo testo italiano per controllare il metodo']*3
        lp,counts=language_model(texts,False)
        prior=dict(log_probabilities=lp,letter_counts=counts,chunks=chunk_counts(texts))
        a=solve('abcabcabca',prior,restarts=1,steps=20,seed=7)
        b=solve('abcabcabca',prior,restarts=1,steps=20,seed=7)
        self.assertEqual(a['recovered'],b['recovered'])
        self.assertTrue(a['candidates']);self.assertEqual(a['selected'],b['selected'])
