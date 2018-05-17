from __future__ import print_function
import numpy as np
import random
from multiprocessing import Process
from openrec.utils.samplers import Sampler

class _MusicSampler(Process):

    def __init__(self, dataset, batch_size, q, genre_f):
        self._dataset = dataset
        self._dataset.shuffle()
        self._batch_size = batch_size
        self._q = q
        self._state = 0
        super(_GeneralSampler, self).__init__()

    def run(self):
        while True:
            
            input_npy = np.zeros(self._batch_size, dtype=[('song_id',np.object, 6000), 
                                                ('source_type', np.object, 6000),
                                                ('source_system_tab', np.object, 6000),
                                                ('source_screen_name', np.object, 6000),
                                                ('artist', np.object, 6000),
                                                ('genre', np.object, 6000),
                                                ('language', np.object, 6000),
                                                ('lyricist', np.object, 6000),
                                                ('composer', np.object, 6000),
                                                ('imp_song_id',np.object), 
                                                ('imp_artist', np.object),
                                                ('imp_genre', np.object),
                                                ('imp_language', np.object),
                                                ('imp_lyricist', np.object),
                                                ('imp_composer', np.object),
                                                ('imp_source_type', np.object),
                                                ('imp_labels',np.object)])

            if self._state + self._batch_size >= len(self._dataset.data):
                self._state = 0
                self._dataset.shuffle()

            for sample_itr, entry in enumerate(self._dataset.data[self._state:(self._state + self._batch_size)]):
                input_npy[sample_itr] = (entry['song_id'],  entry['source_type'],  entry['source_system_tab'], entry['source_screen_name'],
                                        entry['artist'], entry['genre'], entry['language'], entry['lyricist'], entry['composer'],
                                        entry['imp_artist'], entry['imp_genre'], entry['imp_language'], entry['imp_lyricist'], entry['imp_composer'],
                                        entry['imp_source_type'], entry['imp_labels'])
            self._state += self._batch_size
            self._q.put(input_npy, block=True)


class MusicSampler(Sampler):

    def __init__(self, dataset, batch_size, genre_f, chronological=False, num_process=5):
        
        self._chronological = chronological
        if chronological:
            num_process = 1
        
        super(MusicSampler, self).__init__(dataset=dataset, batch_size=batch_size, num_process=num_process)

    def _get_runner(self):
        
        return _MusicSampler(dataset=self._dataset,
                               batch_size=self._batch_size,
                               q=self._q)
