from openrec.recommenders import Recommender
from openrec.modules.extractions import LatentFactor
from openrec.modules.fusions import Average
import tensorflow as tf
from openrec.modules.fusions import Concat
from openrec.modules.interactions.pointwise_mlp_ce import PointwiseMLPCE



class YoutubeRec(Recommender):

    def __init__(self, batch_size, max_user, max_item, dim_embed, 
        test_batch_size=None, l2_reg=None, opt='SGD', lr=None, init_dict=None, sess_config=None):

        self._dim_embed = dim_embed
        
        super(YoutubeRec, self).__init__(batch_size=batch_size, 
                                  test_batch_size=test_batch_size,
                                  max_user=max_user, 
                                  max_item=max_item, 
                                  l2_reg=l2_reg,
                                  opt=opt,
                                  lr=lr,
                                  init_dict=init_dict,
                                  sess_config=sess_config)

    def _build_user_inputs(self, train=True):

        if train:
            self._add_input(name='song_id', dtype='int32', shape=[self._batch_size, 110])
            self._add_input(name='artist', dtype='int32', shape=[self._batch_size, 110])
            self._add_input(name='genre', dtype='int32', shape=[self._batch_size, 110])
            self._add_input(name='language', dtype='int32', shape=[self._batch_size, 110])
            self._add_input(name='lyricist', dtype='int32', shape=[self._batch_size, 110])
            self._add_input(name='composer', dtype='int32', shape=[self._batch_size, 110 ])
        else:
            self._add_input(name='song_id', dtype='int32', shape=[self._batch_size, 110], train = False)
            self._add_input(name='artist', dtype='int32', shape=[self._batch_size, 110], train = False)
            self._add_input(name='genre', dtype='int32', shape=[self._batch_size, 110], train = False)
            self._add_input(name='language', dtype='int32', shape=[self._batch_size, 110], train = False)
            self._add_input(name='lyricist', dtype='int32', shape=[self._batch_size, 110], train = False)
            self._add_input(name='composer', dtype='int32', shape=[self._batch_size, 110 ], train = False)


    def _build_item_inputs(self, train=True):

        if train:
            self._add_input(name='imp_song_id', dtype='int32', shape=[self._batch_size, 1])
            self._add_input(name='imp_artist', dtype='int32', shape=[self._batch_size, 1])
            self._add_input(name='imp_genre', dtype='int32', shape=[self._batch_size, 1])
            self._add_input(name='imp_language', dtype='int32', shape=[self._batch_size, 1])
            self._add_input(name='imp_lyricist', dtype='int32', shape=[self._batch_size, 1])
            self._add_input(name='imp_composer', dtype='int32', shape=[self._batch_size, 1])
        else:
            self._add_input(name='imp_song_id', dtype='int32', shape=[self._batch_size, 1], train = False)
            self._add_input(name='imp_artist', dtype='int32', shape=[self._batch_size, 1], train = False)
            self._add_input(name='imp_genre', dtype='int32', shape=[self._batch_size, 1], train = False)
            self._add_input(name='imp_language', dtype='int32', shape=[self._batch_size, 1], train = False)
            self._add_input(name='imp_lyricist', dtype='int32', shape=[self._batch_size, 1], train = False)
            self._add_input(name='imp_composer', dtype='int32', shape=[self._batch_size, 1], train = False)

    def _build_extra_inputs(self, train=True):
        
        if train:
            self._add_input(name='labels', dtype='int32', shape=[self._batch_size, 110])
            self._add_input(name='source_system_tab', dtype='int32', shape=[self._batch_size, 110])
            self._add_input(name='source_screen_name', dtype='int32', shape=[self._batch_size, 110])
            self._add_input(name='source_type', dtype='int32', shape=[self._batch_size, 110])
        else:
            self._add_input(name='labels', dtype='int32', shape=[self._batch_size], train = False)
            self._add_input(name='source_system_tab', dtype='int32', shape=[self._batch_size, 110], train = False)
            self._add_input(name='source_screen_name', dtype='int32', shape=[self._batch_size, 110], train = False)
            self._add_input(name='source_type', dtype='int32', shape=[self._batch_size, 110], train = False)

    def _input_mappings(self, batch_data, train):

        if train:
           return {

            self._get_input('imp_song_id'): batch_data['imp_song_id_input'],
            self._get_input('imp_artist'): batch_data['imp_artist_input'],
            self._get_input('imp_genre'): batch_data['imp_genre_input'],
            self._get_input('imp_language'): batch_data['imp_language_input'],
            self._get_input('imp_lyricist'): batch_data['imp_lyricist_input'],
            self._get_input('imp_composer'): batch_data['imp_composer_input'],

            self._get_input('song_id'): batch_data['song_id_input'],
            self._get_input('artist'): batch_data['artist_input'],
            self._get_input('genre'): batch_data['genre_input'],
            self._get_input('language'): batch_data['language_input'],
            self._get_input('lyricist'): batch_data['lyricist_input'],
            self._get_input('composer'): batch_data['composer_input'],

            self._get_input('labels'): batch_data['labels_input'],
            self._get_input('source_system_tab'): batch_data['source_system_tab_input'],
            self._get_input('source_screen_name'): batch_data['source_screen_name_input'],
            self._get_input('source_type'): batch_data['source_type_input']}

        else:

            return {self._get_input('imp_song_id', train=train): batch_data['imp_song_id_input'],
            self._get_input('imp_artist', train=train): batch_data['imp_artist_input'],
            self._get_input('imp_genre', train=train): batch_data['imp_genre_input'],
            self._get_input('imp_language', train=train): batch_data['imp_language_input'],
            self._get_input('imp_lyricist', train=train): batch_data['imp_lyricist_input'],
            self._get_input('imp_composer', train=train): batch_data['imp_composer_input'],

            self._get_input('song_id', train=train): batch_data['song_id_input'],
            self._get_input('artist', train=train): batch_data['artist_input'],
            self._get_input('genre', train=train): batch_data['genre_input'],
            self._get_input('language', train=train): batch_data['language_input'],
            self._get_input('lyricist', train=train): batch_data['lyricist_input'],
            self._get_input('composer', train=train): batch_data['composer_input'],

            self._get_input('labels', train=train): batch_data['labels_input'],
            self._get_input('source_system_tab', train=train): batch_data['source_system_tab_input'],
            self._get_input('source_screen_name', train=train): batch_data['source_screen_name_input'],
            self._get_input('source_type', train=train): batch_data['source_type_input']}

    



    def _build_item_extractions(self, train=True):
        if train:
            self._add_module('imp_song_id_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_song_id', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('imp_artist_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_artist', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('imp_genre_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_genre', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('imp_language_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_language', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('imp_lyricist_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_lyricist', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('imp_composer_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_composer', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
        else:
            self._add_module('imp_song_id_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_song_id', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('imp_artist_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_artist', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('imp_genre_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_genre', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('imp_language_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_language', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('imp_lyricist_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_lyricist', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('imp_composer_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('imp_composer', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)


    def _build_extra_extractions(self, train=True):
        if train:
            self._add_module('labels_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('labels', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('source_system_tab_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('source_system_tab', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('source_screen_name_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('source_screen_name', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('source_type_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('source_type', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=tf.AUTO_REUSE), 
                         train=True)
        else:
            self._add_module('labels_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('labels', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('source_system_tab_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('source_system_tab', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('source_screen_name_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('source_screen_name', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('source_type_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('source_type', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)    




    def _build_user_extractions(self, train=True):

        if train:
            self._add_module('song_id_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('song_id', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('artist_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('artist', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('genre_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('genre', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('language_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('language', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('lyricist_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('lyricist', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=tf.AUTO_REUSE), 
                         train=True)
            self._add_module('composer_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('composer', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=tf.AUTO_REUSE), 
                         train=True)
        else:
            self._add_module('song_id_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('song_id', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=True), 
                         train=False)
            self._add_module('artist_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('artist', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=True), 
                         train=False)
            self._add_module('genre_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('genre', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='item', reuse=True), 
                         train=False)
            self._add_module('language_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('language', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=True), 
                         train=False)
            self._add_module('lyricist_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('lyricist', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=True), 
                         train=False)
            self._add_module('composer_vec',
                         LatentFactor(l2_reg=self._l2_reg, init='normal', ids=self._get_input('composer', train=train),
                                    shape=[self._max_item, self._dim_embed], scope='user', reuse=True), 
                         train=False)
        

    def _build_default_fusions(self, train=True):

        self._add_module('song_vec',
            Concat(scope='item_concat', reuse=False, axis=2,
                                module_list=[self._get_module('song_id_vec'), self._get_module('artist_vec'), self._get_module('genre_vec'),
                                self._get_module('language_vec'),self._get_module('lyricist_vec'), self._get_module('composer_vec')]))
        
        self._add_module('song_history',
            tf.reduce_mean(self._get_module('song_vec').get_outputs()[0], axis=1))
        

        self._add_module('impression',
            Concat(scope='item_concat', reuse=False, axis=2,
                                module_list=[self._get_module('imp_song_id_vec'), self._get_module('imp_artist_vec'), self._get_module('imp_genre_vec'),
                                self._get_module('imp_language_vec'),self._get_module('imp_lyricist_vec'), self._get_module('imp_composer_vec')]))

        self._add_module('context',
            Concat(scope='item_concat', reuse=False, axis=2,
                                module_list=[self._get_module('labels_vec'), self._get_module('source_system_tab_vec'), self._get_module('source_screen_name_vec'),
                                self._get_module('source_type_vec')]))


    def _build_default_interactions(self, train=True):
        
        self._add_module('interaction',
            PointwiseMLPCE(user=self._get_module('song_history'), 
                            item=self._get_module('impression').get_outputs()[0], 
                            dims=[1024, 512,1], extra = self._get_module('context').get_outputs()[0], 
                            l2_reg=self._l2_reg, labels=self._get_module('labels_vec').get_outputs()[0]))
        
    def _build_serving_graph(self):
        
        self._scores = self._get_module('interaction', train=False).get_outputs()[0]


