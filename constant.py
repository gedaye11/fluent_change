import os

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
# DEFAULT_WORD_COUNT = 5   # chenged by gexuri

# DEFAULT_WORD_COUNT = 3   # weiboclearV2
DEFAULT_WORD_COUNT = 5   # weiboV2


DEFAULT_LANG = 1  # 0: English, 1: Chinese, ...
DEFAULT_FLUENCY_U = 0.5
TOKEN_PAD = '<pad>'
TOKEN_UNK = '<unk>'
TOKEN_BOS = '<bos>'
# DEFAULT_TRAIN_COLLECTION = 'flickr8kzhbJanbosontrain'  # # changed by gxr
# DEFAULT_TRAIN_COLLECTION = 'flickr30kzhbbosontrain'
# DEFAULT_VAL_COLLECTION = 'flickr30kzhbbosonval'
# DEFAULT_TEST_COLLECTION = 'flickr30kzhmbosontest'
DEFAULT_TRAIN_COLLECTION = ''
DEFAULT_VAL_COLLECTION = ''
DEFAULT_TEST_COLLECTION = ''


DEFAULT_VISUAL_FEAT = 'pyresnet152-pool5osl2'
DEFAULT_MODEL_NAME = '8k_neuraltalk'
DEFAULT_BEAM_SIZE = 10
# DEFAULT_FLUENCY_METHOD = 'sample'
DEFAULT_FLUENCY_METHOD = None


pre_trained_model_path = '/home/gexuri/VisualSearch/flickr30kzhbbosontrain/Models-noScore/fil30k_neuraltalk/vocab_count_thr_5/pyresnet152-pool5osl2/variables/model_47648.ckpt'
# pre_trained_imembedding_path = '/home/gexuri/VisualSearch/flickr30kzhbbosontrain/Models/sample/8k_neuraltalk/vocab_count_thr_5/pyresnet152-pool5osl2/variables/imembedding_model_8619.ckpt'
#pre_trained_lm_path = '/home/gexuri/VisualSearch/flickr30kzhbbosontrain/Models0/sample/8k_neuraltalk/vocab_count_thr_10/pyresnet152-pool5osl2/variables/'
