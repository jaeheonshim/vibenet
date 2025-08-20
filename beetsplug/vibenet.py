from beets.plugins import BeetsPlugin
from beets.library import Library, Item
from vibenet import labels as FIELDS, load_model
from vibenet.core import Model, load_audio
from beets.ui import Subcommand, print_
from beets import ui
import mediafile
import itertools
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

class VibeNetCommand(Subcommand):
    def __init__(self):
        super(VibeNetCommand, self).__init__(
            name='vibenet'
        )
        
    def func(self, lib: Library, opts, args):
        self.lib = lib
        self.query = ui.decargs(args)
        self.net = load_model()
        
        self.predict()
        
    def _process_one(self, item: Item):
        wf = load_audio(item.path.decode(), 16000)
        scores = self.net.predict([wf], 16000)[0]
        scores = scores.to_dict()
        
        # for f in FIELDS:
        #     item[f] = scores[f]
            
        # item.write()
        
        return item
        
    def _show_progress(self, done, total, item: Item):
        print(f"Progress: [{done}/{total}] ({item.artist} - {item.album} - {item.title})", flush=True)
        
    def predict(self):
        items = self.lib.items(self.query)
        total = len(items)
        finished = 0
        
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as ex:
            for it in ex.map(self._process_one, items):
                finished += 1
                self._show_progress(finished, total, it)
            

class VibeNetPlugin(BeetsPlugin):
    def __init__(self):
        super().__init__()
        
        for name in FIELDS:
            field = mediafile.MediaField(
                mediafile.MP3DescStorageStyle(name), mediafile.StorageStyle(name)
            )
            self.add_media_field(name, field)
    
    def commands(self):
        return [VibeNetCommand()]