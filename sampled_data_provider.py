from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import chardet
import pdb
import numpy as np
import os
import logging
import re
import string
import time
import random
import codecs

from constant import ROOT_PATH, DEFAULT_LANG, DEFAULT_FLUENCY_U, TOKEN_PAD, TOKEN_BOS
from bigfile import BigFile
from text import TextTool, TextBank
import utility

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)


class Batch(object):
    def __init__(self, batch_size, max_seq_len, vf_size, bos_ind, 
                 fluency_threshold=DEFAULT_FLUENCY_U):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vf_size = vf_size
        self.bos_ind = bos_ind
        self.fluency_threshold = fluency_threshold
        self.empty()
      
    def empty(self):
        self.x = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
        self.y = np.zeros([self.batch_size, self.max_seq_len], dtype=np.int32)
        self.vf = np.zeros([self.batch_size, self.vf_size], dtype=np.float32)
        self.fg = np.zeros([self.batch_size, self.max_seq_len], dtype=np.float32)
        self.num_feed = 0
        
    def feed_and_vomit(self, visual_feature, sentence):
      i = self.num_feed
      # feed sentence
      self.x[i, 0] = self.bos_ind
      if len(sentence) > self.max_seq_len - 1:
          self.x[i, 1:] = sentence[:self.max_seq_len-1]
          self.y[i, :self.max_seq_len-1] = sentence[:self.max_seq_len-1]
          self.y[i, self.max_seq_len-1] = self.bos_ind
          self.fg[i, :] = np.ones([self.max_seq_len], dtype=np.float32)
      else:
          l = len(sentence)
          self.x[i, 1:l+1] = sentence
          self.y[i, :l] = sentence
          self.y[i, l] = self.bos_ind
          self.fg[i, :l+1] = np.ones([l+1], dtype=np.float32)

      # feed visual feature
      assert visual_feature.shape[0] == self.vf_size
      self.vf[i, :] = visual_feature
      self.num_feed += 1
      assert self.num_feed <= self.batch_size
      # vomit if necessary
      if self.num_feed == self.batch_size:
          return (self.x, self.y, self.vf, self.fg)
      return None


class BucketDataProvider(object):
    """TensorFlow Data Provider with Buckets"""
    def __init__(self, collection, vocab_file, feature, language,
                flag_shuffle=False,  fluency_threshold=DEFAULT_FLUENCY_U, rootpath=ROOT_PATH):
        self.language = language
        self.anno_file_path = utility.get_sent_file(collection, language, rootpath)
        self.fluency_threshold = fluency_threshold
        self.textbank = TextBank(vocab_file)
        assert self.textbank.vocab[TOKEN_PAD] == 0
        self.vf_reader = BigFile(utility.get_feat_dir(collection, feature, rootpath))
        self.vf_names = set(self.vf_reader.names)
        self.vf_size = self.vf_reader.ndims
        self.flag_shuffle = flag_shuffle
        self._load_data()

    def shuffle_data_queue(self):
        random.shuffle(self._data_queue)

    def generate_batches(self, batch_size, buckets):
        """Return a list generator of mini-batches of training data."""
        # create Batches
        batches = []
        for max_seq_len in buckets:
            batches.append(Batch(batch_size, max_seq_len, self.vf_size, self.textbank.vocab[TOKEN_BOS]))
        
        # shuffle if necessary
        if self.flag_shuffle:
            np.random.shuffle(self._data_queue)
        # scan data queue
        for data in self._data_queue:
            # pdb.set_trace()
            sentence = data['sentence']
            # Load visual features
            # print(len(data['image_id']))
            visual_features = np.array(self.vf_reader.read_one(data['image_id']))
            #print("11111111")
            # print (data['image_id'])
            # print(visual_features)
            # print(data['sentence'])
            # sent = self.textbank.decode_tokens(data['sentence'], flag_remove_bos=True)
            # for word in sent:
            #     print (word)
            # # pdb.set_trace()
            if len(sentence) >= buckets[-1]:
                feed_res = batches[-1].feed_and_vomit(visual_features, sentence)
                ind_buc = len(buckets) - 1
            else:
                for (ind_b, batch) in enumerate(batches):
                    if len(sentence) < batch.max_seq_len:
                        feed_res = batches[ind_b].feed_and_vomit(visual_features, sentence)
                        ind_buc = ind_b
                        break
            if feed_res:
                yield (ind_buc,) + feed_res
                batches[ind_buc].empty()

            
    def _load_data(self, verbose=True):
        logger.debug('Loading data')
        self._data_queue = []
        annoss = codecs.open(self.anno_file_path,'r','utf-8').readlines()
        annos = [an.encode('utf-8').decode('utf-8-sig') for an in annoss]

        for (ind_a, line) in enumerate(annos):
            data = {}
            sid, sent = line.strip().split(" ", 1)
            imgid = sid.strip().split("#", 1)[0]
            # print(imgid)
            assert(imgid in self.vf_names)
            # pdb.set_trace()
            # if imgid not in self.vf_names:
            #    print(imgid)
            #    logger.info('%s not in feature data, skipping that.'%imgid)
            #    pdb.set_trace()
            #    continue
            data['image_id'] = imgid
            # print(imgid)
            # # Encode sentences

            tokens = TextTool.tokenize(sent, self.language)
            data['sentence'] = self.textbank.encode_tokens(tokens, flag_add_bos=False)
            self._data_queue.append(data)
            if verbose and (ind_a + 1) % 20000 == 0:
                logger.debug('%d/%d annotation', ind_a + 1, len(annos))
        random.shuffle( self._data_queue )   #       ############################# changed by gxr
        
        nr_of_images = len(set([data['image_id'] for data in self._data_queue]))
        logger.info('%d images, %d sentences from %s', nr_of_images, len(self._data_queue), self.anno_file_path)

if __name__ == '__main__':
    from utility import get_vocab_file
    rootpath = ROOT_PATH
    # collection = 'flickr8kenctrain'
    # collection = 'flickr8kzhbJanbosontrain'
    #collection = 'flickr8kzh'
    collection = 'weibotrain'
    word_cnt_thr = 5
    feature = 'pyresnet152-pool5osl2'
    data_provider = BucketDataProvider(collection, get_vocab_file(collection), feature, language=1, rootpath=rootpath)
    batch_size = 100
    buckets = [16]

    for step, (ind_buc, x, y, vf, fg) in enumerate(data_provider.generate_batches(batch_size, buckets)):
        print (step, ind_buc, x.shape, vf.shape)
        # print (vf[0:3])
        break
        #print (x[0])
        #break
  

