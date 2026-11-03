"""Finding the feature log for a recording, and the window -g opens to pick one.

Logs are named cf_<video file name>.json[.gz] (see main._feature_log_path), and -r moves the ones
whose video is gone into an old/ directory beside them, so a recording's log is found by the
video's name in each log directory and its old/.
"""
import os
import re

_LOG_NAME = re.compile(r'^cf_(.+)\.[A-Za-z0-9]{2,4}\.json(?:\.gz)?$')

def log_dirs(opts) -> list[str]:
    """Where to look for feature logs: --log-dir, else $PYCOMMFLAG_LOG_DIR and the temp dir
    (where a flag run keeps its log by default); each followed by its old/."""
    import tempfile
    base = list(getattr(opts, 'log_dir', None) or [])
    if not base:
        if d := os.environ.get('PYCOMMFLAG_LOG_DIR'):
            base.append(d)
        base.append(tempfile.gettempdir())
    out = []
    for d in base:
        for x in (d, os.path.join(d, 'old')):
            x = os.path.abspath(x)
            if os.path.isdir(x) and x not in out:
                out.append(x)
    return out

def _stem(name:str) -> str:
    return name.rsplit('.', 1)[0] if '.' in name else name

def find_log(stem:str, dirs:list[str], exts=('ts', 'mkv')) -> str|None:
    """The first cf_<stem>.<ext>.json[.gz] in `dirs`, in their order."""
    for d in dirs:
        for ext in exts:
            for suffix in ('.json.gz', '.json'):
                p = os.path.join(d, f'cf_{stem}.{ext}{suffix}')
                if os.path.isfile(p):
                    return p
    return None

def resolve_gui_arg(arg:str|None, dirs:list[str]) -> tuple[str|None, str|None, str|None]:
    """What -g was given, as (log, video, search text).

    A feature log is used as is.  A video, or a link to one (mythlink's), means its log, found by
    the name the link points at.  A <chanid>_<starttime> means that recording's log.  Anything
    else (including nothing) is text for the picker to filter on, and comes back as the third
    item with no log.  Raises LookupError when it names a recording whose log can't be found.
    """
    if not arg:
        return (None, None, '')
    if re.search(r'\.json(\.gz)?$', arg):
        if not os.path.isfile(arg):
            raise LookupError(f'No such feature log "{arg}"')
        return (arg, None, None)
    if os.path.isfile(arg) or os.path.islink(arg):
        # a link can outlive its target (the .ts is transcoded to .mkv), and its name is still the key
        video = os.path.realpath(arg)
        name = os.path.basename(video)
        ext = name.rsplit('.', 1)[1] if '.' in name else ''
        exts = tuple(dict.fromkeys(e for e in (ext, 'ts', 'mkv') if e))
        if log := find_log(_stem(name), dirs, exts):
            return (log, video if os.path.isfile(video) else None, None)
        raise LookupError(f'No feature log for {video} (looked for cf_{_stem(name)}.* in {", ".join(dirs)})')
    if re.fullmatch(r'\d{4,6}_\d{12,}', arg):
        if log := find_log(arg, dirs):
            return (log, None, None)
        raise LookupError(f'No feature log for {arg} (looked in {", ".join(dirs)})')
    if os.path.sep in arg:
        raise LookupError(f'No such file "{arg}"')
    return (None, None, arg)

def read_header(path:str, size:int=4096) -> dict:
    """`filename` and `edited` from the start of a feature log, without parsing the whole thing
    (processor.set_header_key puts `edited` up there)."""
    import json
    if path.endswith('.gz'):
        import gzip
        f = gzip.open(path, 'rt')
    else:
        f = open(path, 'r')
    with f:
        head = f.read(size)
    out = {}
    for key in ('filename', 'edited'):
        m = re.search(r'"' + key + r'"\s*:\s*("(?:[^"\\]|\\.)*")', head)
        out[key] = json.loads(m[1]) if m else None
    return out

def log_index(dirs:list[str]) -> dict[str, str]:
    """{stem: log path} for every log in `dirs`; a stem in more than one goes to the earliest dir."""
    out = {}
    for d in dirs:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            if (m := _LOG_NAME.match(n)) and m[1] not in out:
                out[m[1]] = os.path.join(d, n)
    return out

class _Row:
    __slots__ = ('stem', 'when', 'date', 'title', 'subtitle', 'group', 'log', 'where', 'edited')

def _rows(dirs:list[str]) -> list[_Row]:
    """A row per recording MythTV knows about (or, without MythTV, per log), newest first.
    `edited` is left None: reading it is a file open per log (see _read_edited)."""
    from datetime import datetime, timezone
    from . import mythtv

    logs = log_index(dirs)
    rows = []
    recs = mythtv.list_recordings()
    if recs is not None:
        for (chanid, start, title, subtitle, basename, group) in recs:
            r = _Row()
            r.stem = _stem(basename or f'{chanid}_{start:%Y%m%d%H%M%S}')
            r.when = start or datetime.min
            # the database's times are UTC
            r.date = start.replace(tzinfo=timezone.utc).astimezone().strftime('%Y-%m-%d %a %H:%M') if start else ''
            (r.title, r.subtitle, r.group) = (title or '', subtitle or '', group or '')
            rows.append(r)
    else:
        for stem in logs:
            r = _Row()
            r.stem = stem
            try:
                r.when = datetime.strptime(stem.split('_', 1)[-1][:14], '%Y%m%d%H%M%S')
            except ValueError:
                r.when = datetime.min
            (r.date, r.title, r.subtitle, r.group) = ('', stem, '', '')
            rows.append(r)
    for r in rows:
        r.log = logs.get(r.stem)
        r.where = '' if r.log is None else 'old' if os.path.basename(os.path.dirname(r.log)) == 'old' else 'log'
        r.edited = None
    rows.sort(key=lambda r: r.when, reverse=True)
    return rows

def _read_edited(rows:list[_Row], out, stop):
    """Put (row index, edited or '') on `out` for each row with a log, in order, until `stop` is set.
    A cold cache of a RAID full of logs made this take over a minute, so it runs on a thread."""
    for (i, r) in enumerate(rows):
        if stop.is_set():
            return
        if r.log:
            try:
                out.put((i, read_header(r.log).get('edited') or ''))
            except Exception:
                out.put((i, ''))

def pick(dirs:list[str], text:str='') -> str|None:
    """Show the recordings, filtered by `text` to start with, and return the chosen one's log (None if closed)."""
    import tkinter as tk
    from tkinter import ttk

    import queue
    import threading

    rows = _rows(dirs)
    to_read = sum(1 for r in rows if r.log)
    edited = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=_read_edited, args=(rows, edited, stop), daemon=True).start()
    root = tk.Tk()
    # which directories it found logs in is the first thing to check when a log seems missing
    root.title('pycommflag: pick a recording (logs from ' + ', '.join(d for d in dirs if os.path.basename(d) != 'old') + ')')
    filt = tk.StringVar(value=text)
    only_log = tk.BooleanVar(value=True)
    hide_edited = tk.BooleanVar(value=False)
    chosen = []

    bar = tk.Frame(root)
    bar.grid(row=0, column=0, columnspan=2, sticky='we', padx=5, pady=5)
    tk.Label(bar, text='Filter:').pack(side=tk.LEFT)
    entry = tk.Entry(bar, textvariable=filt, width=40)
    entry.pack(side=tk.LEFT, padx=5)
    tk.Checkbutton(bar, text='only with a log', variable=only_log).pack(side=tk.LEFT, padx=5)
    tk.Checkbutton(bar, text='hide edited', variable=hide_edited).pack(side=tk.LEFT, padx=5)
    status = tk.Label(bar)
    status.pack(side=tk.RIGHT)

    cols = (('date', 'Recorded', 160), ('title', 'Title', 280), ('subtitle', 'Episode', 300),
            ('group', 'Group', 120), ('log', 'Log', 50), ('edited', 'Edited', 150))
    # ttk's row height is a fixed number of pixels, not the font's: under any scaling larger than
    # Tk's default (a HiDPI screen, a forwarded X display) each row shows only half its text
    import tkinter.font as tkfont
    style = ttk.Style(root)
    line = max(tkfont.nametofont(f).metrics('linespace') for f in ('TkDefaultFont', 'TkTextFont'))
    style.configure('Treeview', rowheight=line + 6)
    tree = ttk.Treeview(root, columns=[c for (c,_,_) in cols], show='headings', height=30, selectmode='browse')
    for (c, name, w) in cols:
        tree.heading(c, text=name)
        tree.column(c, width=w, stretch=(c in ('title', 'subtitle')))
    tree.tag_configure('nolog', foreground='grey50')
    scroll = ttk.Scrollbar(root, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.grid(row=1, column=0, sticky='nswe')
    scroll.grid(row=1, column=1, sticky='ns')
    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)

    def refill(*_):
        tree.delete(*tree.get_children())
        words = filt.get().lower().split()
        for (i, r) in enumerate(rows):
            if (only_log.get() and not r.log) or (hide_edited.get() and r.edited):
                continue
            hay = f'{r.date} {r.title} {r.subtitle} {r.group} {r.stem}'.lower()
            if all(w in hay for w in words):
                tree.insert('', 'end', iid=str(i), tags=() if r.log else ('nolog',),
                            values=(r.date, r.title, r.subtitle, r.group, r.where or '-',
                                    '...' if r.edited is None and r.log else r.edited or ''))
        kids = tree.get_children()
        show_status()
        if kids:
            tree.focus(kids[0])
            tree.selection_set(kids[0])

    read = [0]
    def show_status():
        text = f'{len(tree.get_children())} of {len(rows)} recordings'
        if read[0] < to_read:
            text += f' (checking which are edited: {read[0]} of {to_read})'
        status.configure(text=text)

    job = [None]
    def close(_=None):
        # cancelled first: a pending after() fires "invalid command name" once the window is gone
        if stop.is_set():
            return # already closed (a second Enter or double-click before the first one's close ran)
        stop.set()
        if job[0] is not None:
            root.after_cancel(job[0])
            job[0] = None
        root.destroy()

    def poll():
        # the edited column, as _read_edited gets to each log
        job[0] = None
        try:
            while True:
                (i, value) = edited.get_nowait()
                rows[i].edited = value
                read[0] += 1
                if tree.exists(str(i)):
                    if value and hide_edited.get():
                        tree.delete(str(i))
                    else:
                        tree.set(str(i), 'edited', value)
        except queue.Empty:
            pass
        show_status()
        if read[0] < to_read:
            job[0] = root.after(200, poll)

    def open_row(_=None):
        # the window closes once this event is done with: destroyed in here, the treeview's own
        # double-click handling still ran after it (on a widget that was gone), 'break' or not
        sel = tree.selection()
        if not sel:
            return 'break'
        r = rows[int(sel[0])]
        if not r.log:
            root.bell()
            return 'break'
        chosen.append(r.log)
        root.after_idle(close)
        return 'break'

    def from_entry(_=None):
        kids = tree.get_children()
        if len(kids) == 1:
            open_row()
        elif kids:
            tree.focus_set()
            tree.focus(kids[0])

    for v in (filt, only_log, hide_edited):
        v.trace_add('write', refill)
    tree.bind('<Double-1>', open_row)
    tree.bind('<Return>', open_row)
    entry.bind('<Return>', from_entry)
    entry.bind('<Down>', from_entry)
    root.bind('<Escape>', close)
    root.protocol('WM_DELETE_WINDOW', close)
    refill()
    poll()
    entry.focus_set()
    entry.icursor(tk.END)
    root.mainloop()
    return chosen[0] if chosen else None
