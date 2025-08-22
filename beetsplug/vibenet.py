import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from optparse import OptionParser

import mediafile
from beets import ui
from beets.library import Item, Library
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, print_

from vibenet import labels as FIELDS
from vibenet import load_model
from vibenet.core import Model, load_audio


class VibeNetCommand(Subcommand):
    cfg_dry_run = False
    cfg_threads = 0
    cfg_force = False
    
    def __init__(self, config):
        self.config = config
        
        cfg = self.config.flatten()
        self.cfg_auto = cfg.get("auto")
        self.cfg_threads = cfg.get("threads")
        self.cfg_force = cfg.get("force")
                
        self.parser = OptionParser()
        
        self.parser.add_option(
            '-d', '--dry-run',
            action='store_true', dest='dryrun'
        )
        
        self.parser.add_option(
            '-t', '--threads',
            action='store', dest='threads', type='int',
            default=self.cfg_threads
        )
        
        self.parser.add_option(
            '-f', '--force',
            action='store_true', dest='force',
            default=self.cfg_force
        )
        
        super(VibeNetCommand, self).__init__(
            parser=self.parser,
            name='vibenet'
        )
        
    def func(self, lib: Library, opts, args):
        self.cfg_dry_run = opts.dryrun
        self.cfg_threads = opts.threads
        self.cfg_force = opts.force
                
        if not self.cfg_threads or self.cfg_threads == 0:
            self.cfg_threads = multiprocessing.cpu_count()
            print("Adjusting max threads to CPU count: {0}".format(self.cfg_threads))
        
        self.lib = lib
        self.query = ui.decargs(args)
        self.net = load_model()
        
        if self.cfg_dry_run:
            print("*******************************************")
            print("*** DRY RUN: NO CHANGES WILL BE APPLIED ***")
            print("*******************************************")
        
        self.predict()
        
    def _process_one(self, item: Item):
        wf = load_audio(item.path.decode(), 16000)
        scores = self.net.predict([wf], 16000)[0]
        scores = scores.to_dict()
        
        if not self.cfg_dry_run:
            for f in FIELDS:
                item[f] = scores[f]
            
            item.write()
        
        return item
        
    def _show_progress(self, done, total, item: Item):
        print(f"Progress: [{done}/{total}] ({item.artist} - {item.album} - {item.title})", flush=True)
        
    def predict(self):
        items = self.lib.items(self.query)
        total = len(items)
        finished = 0
        
        with ThreadPoolExecutor(max_workers=self.cfg_threads) as ex:
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
        return [VibeNetCommand(self.config)]