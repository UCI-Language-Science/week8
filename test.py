import os
import numpy as np
import pandas as pd

from nltk.corpus import treebank

# Import the implemented functions
from analysis import (
    tagged_sentences,
    get_pos_tag_observations,
    get_n_most_common_pos_tags,
    rarest_pos_tag_word,
    get_heights,
    get_leaves,
    predict_height,
    plot_10_most_common_pos_tags,
    plot_regression,    
)

toy_corpus = "The replicator equation is interpreted as a continuous inference equation and a formal similarity between the discrete replicator equation and Bayesian inference is described. Further connections between inference and the replicator equation are given including a discussion of information divergences, evolutionary stability, and exponential families as solutions for the replicator dynamic, using Fisher information and information geometry.".split(".")

full_corpus = treebank.sents()
full_corpus = [' '.join(sent) for sent in full_corpus]
short_corpus = full_corpus[:10]

# Part I.

class TestTaggedSentences:

    def test_toy(self):

        expected = [[('The', 'DT'), ('replicator', 'NN'), ('equation', 'NN'), ('is', 'VBZ'), ('interpreted', 'VBN'), ('as', 'IN'), ('a', 'DT'), ('continuous', 'JJ'), ('inference', 'NN'), ('equation', 'NN'), ('and', 'CC'), ('a', 'DT'), ('formal', 'JJ'), ('similarity', 'NN'), ('between', 'IN'), ('the', 'DT'), ('discrete', 'JJ'), ('replicator', 'NN'), ('equation', 'NN'), ('and', 'CC'), ('Bayesian', 'JJ'), ('inference', 'NN'), ('is', 'VBZ'), ('described', 'VBN')], [('Further', 'JJ'), ('connections', 'NNS'), ('between', 'IN'), ('inference', 'NN'), ('and', 'CC'), ('the', 'DT'), ('replicator', 'NN'), ('equation', 'NN'), ('are', 'VBP'), ('given', 'VBN'), ('including', 'VBG'), ('a', 'DT'), ('discussion', 'NN'), ('of', 'IN'), ('information', 'NN'), ('divergences', 'NNS'), (',', ','), ('evolutionary', 'JJ'), ('stability', 'NN'), (',', ','), ('and', 'CC'), ('exponential', 'JJ'), ('families', 'NNS'), ('as', 'IN'), ('solutions', 'NNS'), ('for', 'IN'), ('the', 'DT'), ('replicator', 'NN'), ('dynamic', 'NN'), (',', ','), ('using', 'VBG'), ('Fisher', 'NNP'), ('information', 'NN'), ('and', 'CC'), ('information', 'NN'), ('geometry', 'NN')], []]
        actual = tagged_sentences(toy_corpus)
        assert expected == actual

    
    def test_short(self):

        actual = tagged_sentences(short_corpus)

        # Test the last entry
        last_expected = [('There', 'EX'), ('is', 'VBZ'), ('no', 'DT'), ('asbestos', 'NN'), ('in', 'IN'), ('our', 'PRP$'), ('products', 'NNS'), ('now', 'RB'), ('.', '.'), ('``', '``')]
        assert last_expected == actual[-1]

        # Test the first entry
        first_expected = [('Pierre', 'NNP'), ('Vinken', 'NNP'), (',', ','), ('61', 'CD'), ('years', 'NNS'), ('old', 'JJ'), (',', ','), ('will', 'MD'), ('join', 'VB'), ('the', 'DT'), ('board', 'NN'), ('as', 'IN'), ('a', 'DT'), ('nonexecutive', 'JJ'), ('director', 'NN'), ('Nov.', 'NNP'), ('29', 'CD'), ('.', '.')]
        assert first_expected == actual[0]

        # Test a middle entry
        middle_expected = [('The', 'DT'), ('asbestos', 'NN'), ('fiber', 'NN'), (',', ','), ('crocidolite', 'NN'), (',', ','), ('is', 'VBZ'), ('unusually', 'RB'), ('resilient', 'JJ'), ('once', 'IN'), ('it', 'PRP'), ('enters', 'VBZ'), ('the', 'DT'), ('lungs', 'NNS'), (',', ','), ('with', 'IN'), ('even', 'RB'), ('brief', 'JJ'), ('exposures', 'NNS'), ('to', 'TO'), ('it', 'PRP'), ('causing', 'VBG'), ('symptoms', 'NNS'), ('that', 'WDT'), ('*', 'VBP'), ('T', 'NNP'), ('*', 'NNP'), ('-1', 'NNP'), ('show', 'VBP'), ('up', 'RP'), ('decades', 'NNS'), ('later', 'RB'), (',', ','), ('researchers', 'NNS'), ('said', 'VBD'), ('0', 'CD'), ('*', 'NNP'), ('T', 'NNP'), ('*', 'NNP'), ('-2', 'NNP'), ('.', '.')]
        assert middle_expected == actual[4]

    def test_full(self):

        # Test an arbitrary index
        actual = tagged_sentences(full_corpus)

        expected = ('ago', 'RB')
        assert expected == actual[-1000][3]

        expected = ('farmers', 'NNS')
        assert expected == actual[2986][8]

        expected = ('*', 'JJ')
        assert expected == actual[500][7]


class TestDataFrame:

    def test_df_toy(self):
        expected_dict = {'word': {0: 'The', 1: 'replicator', 2: 'equation', 3: 'is', 4: 'interpreted', 5: 'as', 6: 'a', 7: 'continuous', 8: 'inference', 9: 'equation', 10: 'and', 11: 'a', 12: 'formal', 13: 'similarity', 14: 'between', 15: 'the', 16: 'discrete', 17: 'replicator', 18: 'equation', 19: 'and', 20: 'Bayesian', 21: 'inference', 22: 'is', 23: 'described', 24: 'Further', 25: 'connections', 26: 'between', 27: 'inference', 28: 'and', 29: 'the', 30: 'replicator', 31: 'equation', 32: 'are', 33: 'given', 34: 'including', 35: 'a', 36: 'discussion', 37: 'of', 38: 'information', 39: 'divergences', 40: ',', 41: 'evolutionary', 42: 'stability', 43: ',', 44: 'and', 45: 'exponential', 46: 'families', 47: 'as', 48: 'solutions', 49: 'for', 50: 'the', 51: 'replicator', 52: 'dynamic', 53: ',', 54: 'using', 55: 'Fisher', 56: 'information', 57: 'and', 58: 'information', 59: 'geometry'}, 'pos': {0: 'DT', 1: 'NN', 2: 'NN', 3: 'VBZ', 4: 'VBN', 5: 'IN', 6: 'DT', 7: 'JJ', 8: 'NN', 9: 'NN', 10: 'CC', 11: 'DT', 12: 'JJ', 13: 'NN', 14: 'IN', 15: 'DT', 16: 'JJ', 17: 'NN', 18: 'NN', 19: 'CC', 20: 'JJ', 21: 'NN', 22: 'VBZ', 23: 'VBN', 24: 'JJ', 25: 'NNS', 26: 'IN', 27: 'NN', 28: 'CC', 29: 'DT', 30: 'NN', 31: 'NN', 32: 'VBP', 33: 'VBN', 34: 'VBG', 35: 'DT', 36: 'NN', 37: 'IN', 38: 'NN', 39: 'NNS', 40: ',', 41: 'JJ', 42: 'NN', 43: ',', 44: 'CC', 45: 'JJ', 46: 'NNS', 47: 'IN', 48: 'NNS', 49: 'IN', 50: 'DT', 51: 'NN', 52: 'NN', 53: ',', 54: 'VBG', 55: 'NNP', 56: 'NN', 57: 'CC', 58: 'NN', 59: 'NN'}, 'word_num': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 0, 25: 1, 26: 2, 27: 3, 28: 4, 29: 5, 30: 6, 31: 7, 32: 8, 33: 9, 34: 10, 35: 11, 36: 12, 37: 13, 38: 14, 39: 15, 40: 16, 41: 17, 42: 18, 43: 19, 44: 20, 45: 21, 46: 22, 47: 23, 48: 24, 49: 25, 50: 26, 51: 27, 52: 28, 53: 29, 54: 30, 55: 31, 56: 32, 57: 33, 58: 34, 59: 35}, 'sent_num': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1, 31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1, 40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1, 50: 1, 51: 1, 52: 1, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1}}
        expected = pd.DataFrame(expected_dict).to_numpy()
        actual = get_pos_tag_observations(toy_corpus).to_numpy()
        assert np.array_equal(expected, actual)

    def test_df_short(self):
        
        expected_dict = {'word': {150: 'New', 151: 'York-based', 152: 'Loews', 153: 'Corp.', 154: 'that', 155: '*', 156: 'T', 157: '*', 158: '-2', 159: 'makes', 160: 'Kent', 161: 'cigarettes', 162: ',', 163: 'stopped', 164: 'using', 165: 'crocidolite', 166: 'in', 167: 'its', 168: 'Micronite', 169: 'cigarette', 170: 'filters', 171: 'in', 172: '1956', 173: '.', 174: 'Although', 175: 'preliminary', 176: 'findings', 177: 'were', 178: 'reported', 179: '*', 180: '-2', 181: 'more', 182: 'than', 183: 'a', 184: 'year', 185: 'ago', 186: ',', 187: 'the', 188: 'latest', 189: 'results', 190: 'appear', 191: 'in', 192: 'today', 193: "'s", 194: 'New', 195: 'England', 196: 'Journal', 197: 'of', 198: 'Medicine', 199: ','}, 'pos': {150: 'NNP', 151: 'JJ', 152: 'NNP', 153: 'NNP', 154: 'IN', 155: 'NNP', 156: 'NNP', 157: 'NNP', 158: 'NNP', 159: 'VBZ', 160: 'NNP', 161: 'NNS', 162: ',', 163: 'VBD', 164: 'VBG', 165: 'NN', 166: 'IN', 167: 'PRP$', 168: 'NNP', 169: 'NN', 170: 'NNS', 171: 'IN', 172: 'CD', 173: '.', 174: 'IN', 175: 'JJ', 176: 'NNS', 177: 'VBD', 178: 'VBN', 179: 'JJ', 180: 'NN', 181: 'JJR', 182: 'IN', 183: 'DT', 184: 'NN', 185: 'RB', 186: ',', 187: 'DT', 188: 'JJS', 189: 'NNS', 190: 'VBP', 191: 'IN', 192: 'NN', 193: 'POS', 194: 'NNP', 195: 'NNP', 196: 'NNP', 197: 'IN', 198: 'NNP', 199: ','}, 'word_num': {150: 6, 151: 7, 152: 8, 153: 9, 154: 10, 155: 11, 156: 12, 157: 13, 158: 14, 159: 15, 160: 16, 161: 17, 162: 18, 163: 19, 164: 20, 165: 21, 166: 22, 167: 23, 168: 24, 169: 25, 170: 26, 171: 27, 172: 28, 173: 29, 174: 0, 175: 1, 176: 2, 177: 3, 178: 4, 179: 5, 180: 6, 181: 7, 182: 8, 183: 9, 184: 10, 185: 11, 186: 12, 187: 13, 188: 14, 189: 15, 190: 16, 191: 17, 192: 18, 193: 19, 194: 20, 195: 21, 196: 22, 197: 23, 198: 24, 199: 25}, 'sent_num': {150: 5, 151: 5, 152: 5, 153: 5, 154: 5, 155: 5, 156: 5, 157: 5, 158: 5, 159: 5, 160: 5, 161: 5, 162: 5, 163: 5, 164: 5, 165: 5, 166: 5, 167: 5, 168: 5, 169: 5, 170: 5, 171: 5, 172: 5, 173: 5, 174: 6, 175: 6, 176: 6, 177: 6, 178: 6, 179: 6, 180: 6, 181: 6, 182: 6, 183: 6, 184: 6, 185: 6, 186: 6, 187: 6, 188: 6, 189: 6, 190: 6, 191: 6, 192: 6, 193: 6, 194: 6, 195: 6, 196: 6, 197: 6, 198: 6, 199: 6}}
        expected = pd.DataFrame(expected_dict).to_numpy()
        actual = get_pos_tag_observations(short_corpus).iloc[150:200].to_numpy()
        assert np.array_equal(expected, actual)

    def test_df_full(self):

        expected_dict = {'word': {109781: 'begin', 109782: 'delivery', 109783: 'in', 109784: 'the', 109785: 'first', 109786: 'quarter', 109787: 'of', 109788: 'next', 109789: 'year', 109790: '.'}, 'pos': {109781: 'VB', 109782: 'NN', 109783: 'IN', 109784: 'DT', 109785: 'JJ', 109786: 'NN', 109787: 'IN', 109788: 'JJ', 109789: 'NN', 109790: '.'}, 'word_num': {109781: 8, 109782: 9, 109783: 10, 109784: 11, 109785: 12, 109786: 13, 109787: 14, 109788: 15, 109789: 16, 109790: 17}, 'sent_num': {109781: 3913, 109782: 3913, 109783: 3913, 109784: 3913, 109785: 3913, 109786: 3913, 109787: 3913, 109788: 3913, 109789: 3913, 109790: 3913}}
        expected = pd.DataFrame(expected_dict).to_numpy()
        actual_df = get_pos_tag_observations(full_corpus)
        assert len(actual_df) == 109791

        actual = actual_df.iloc[-10:].to_numpy()
        assert np.array_equal(expected, actual)


class TestNMostCommon:

    def test_n_most_common_toy(self):

        actual = get_n_most_common_pos_tags(toy_corpus, 10)
        expected = ['NN', 'DT', 'JJ', 'IN', 'CC', 'NNS', 'VBN', ',', 'VBZ', 'VBG']
        assert actual == expected

    def test_n_most_common_short(self):

        actual = get_n_most_common_pos_tags(short_corpus, 10)
        expected = ['NNP', 'NN', 'IN', 'NNS', 'DT', 'JJ', ',', '.', 'RB', 'VBZ']
        assert expected == actual

    def test_n_most_common_full(self):

        actual = get_n_most_common_pos_tags(full_corpus, 10)
        expected = ['NNP', 'NN', 'IN', 'DT', 'JJ', 'NNS', ',', 'CD', '.', 'VBD']
        assert expected == actual

    def test_sorted_full(self):

        actual = get_n_most_common_pos_tags(full_corpus, len(full_corpus))
        expected = ['NNP', 'NN', 'IN', 'DT', 'JJ', 'NNS', ',', 'CD', '.', 'VBD', 'RB', 'VB', 'VBZ', 'CC', 'TO', 'VBN', 'VBP', 'PRP', 'VBG', '``', 'MD', 'POS', 'PRP$', '$', ':', 'WDT', 'JJR', 'WP', 'RP', 'JJS', 'WRB', 'NNPS', 'RBR', 'EX', 'RBS', 'PDT', 'FW', "''", '#', 'WP$', 'UH', 'LS', 'SYM']
        assert expected == actual


class TestRarestPosTagWord:

    # The 'rarest' aren't unique, but should still pass the test.

    def test_rarest_pos_tag_word_toy(self):

        actual = rarest_pos_tag_word(toy_corpus)
        expected = "are"
        assert expected == actual
    
    def test_rarest_pos_tag_word_short(self):

        actual = rarest_pos_tag_word(short_corpus)
        expected = "and"
        assert expected == actual

    def test_rarest_pos_tag_word_full(self):

        actual = rarest_pos_tag_word(full_corpus)
        expected = "b"
        assert expected == actual


# Part II.

parsed_sentences = treebank.parsed_sents()

class TestRegression:

    def test_heights(self):
        actual_arr = get_heights(parsed_sentences)
        indices = np.array((99, 46, 13, 1000, 2555, 3131))
        actual = actual_arr[indices]
        expected = np.array([12,  8, 15, 11, 13, 12])
        assert np.array_equal(expected, actual)

    def test_leaves(self):
        actual_arr = get_leaves(parsed_sentences)
        indices = np.array((99, 46, 13, 1000, 2555, 3131))
        actual = actual_arr[indices]
        expected = np.array([47, 29, 38, 29, 16, 24])
        assert np.array_equal(expected, actual)

    def test_predict(self):

        actual = predict_height(parsed_sentences, 1)
        expected = 6.088529855226791
        assert np.isclose(expected, actual)

        actual = predict_height(parsed_sentences, 10)
        expected = 7.924813182563772
        assert np.isclose(expected, actual)

        actual = predict_height(parsed_sentences, 250)
        expected = 56.892368578216626
        assert np.isclose(expected, actual)


class TestPlots:

    # These don't really test anything, but you can use them to call your functions and thus generate the image files.
    def test_plot_pos_tags(self):
        plot_10_most_common_pos_tags(full_corpus)
        assert os.path.exists("histogram.png")
    
    def test_plot_regression(self):
        plot_regression(parsed_sentences)
        assert os.path.exists("regression.png")
