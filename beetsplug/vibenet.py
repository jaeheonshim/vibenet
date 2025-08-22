import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

import mediafile
from beets import ui
from beets.importer import ImportTask
from beets.library import Item, Library
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, should_write
from beets.util import syspath

from vibenet import labels as FIELDS
from vibenet import load_model
from vibenet.core import load_audio


class VibeNetPlugin(BeetsPlugin):
    def __init__(self):
        super().__init__()
        
        self.config.add({
            "threads": 0,
            "auto": True,
            "force": False
        })
        
        self.cfg_threads = self.config['threads'].get(int)
        self.cfg_auto = self.config['auto'].get(bool)
        self.cfg_force = self.config['force'].get(bool)
        
        for name in FIELDS:
            field = mediafile.MediaField(
                mediafile.MP3DescStorageStyle(name), mediafile.StorageStyle(name)
            )
            self.add_media_field(name, field)
            
        self.import_stages = [self.imported]
            
    def _process_items(self, items: list[Item], threads: int, dry_run: bool, write_tags: bool, force: bool):
        if not force:
            # Skip items that already have tags
            items = [it for it in items if any(it.get(f) is None for f in FIELDS)]

        if not threads or threads == 0:
            threads = multiprocessing.cpu_count()
            print("Adjusting max threads to CPU count: {0}".format(threads))

        net = load_model()
        
        def worker(item) -> tuple[Item, dict]:
            path = syspath(item.path)
            wf = load_audio(path, 16000)
            pred = net.predict([wf], 16000)[0]
            scores = pred.to_dict()
            return item, scores

        total = len(items)
        finished = 0
        
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = {ex.submit(worker, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                idx = futs[fut]
                it, scores = fut.result()
                
                if not dry_run:
                    for f in FIELDS:
                        it[f] = float(scores[f])
                        
                    it.store()
                    
                    if write_tags:
                        it.write()
                
                finished += 1
                print(f"Progress: [{finished}/{total}] ({it.artist} - {it.album} - {it.title})", flush=True)
    
    def commands(self):
        cmd = Subcommand("vibenet", help="Predict VibeNet attributes and store them on items.")
        
        cmd.parser.add_option(
            '-d', '--dry-run',
            action='store_true', dest='dryrun',
            help="Run predictions but do not save results to the library or files."
        )
        
        cmd.parser.add_option(
            '-w', '--write',
            action='store_true', dest='write',
            help="Also write predicted attributes into file tags (not just the beets database)."
        )
        
        cmd.parser.add_option(
            '-t', '--threads',
            action='store', dest='threads', type='int',
            default=self.cfg_threads,
            help="Number of worker threads to use for predictions."
        )
        
        cmd.parser.add_option(
            '-f', '--force',
            action='store_true', dest='force',
            default=self.cfg_force,
            help="Recompute and overwrite attributes even if they are already present."
        )
        cmd.func = self._run_cmd
        return [cmd]
        
    def imported(self, _, task: ImportTask) -> None:
        if not self.cfg_auto:
            return
        
        items = list(task.imported_items())
        self._process_items(items, threads=self.cfg_threads, dry_run=False, write_tags=should_write(), force=self.cfg_force)
        
    def _run_cmd(self, lib: Library, opts, args):
        query = ui.decargs(args)
        items = lib.items(query)
        
        if opts.dryrun:
            print("*******************************************")
            print("*** DRY RUN: NO CHANGES WILL BE APPLIED ***")
            print("*******************************************")
            
        self._process_items(items, threads=opts.threads, dry_run=opts.dryrun, write_tags=opts.write, force=opts.force)
        