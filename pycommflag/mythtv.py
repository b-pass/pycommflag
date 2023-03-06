import sys
import os
import xml.etree.ElementTree as xml
import MySQLdb as mysql
import logging as log
import av

def _open():
    dbc = {}
    cfgfile = os.path.join(os.path.expanduser('~'), '.mythtv/config.xml')
    for e in xml.parse(cfgfile).find('Database').iter():
        dbc[e.tag.lower()] = e.text
    return mysql.connect(
        host=dbc.get('host', "localhost"),
        user=dbc.get('username', "mythtv"),
        passwd=dbc.get('password', "mythtv"),
        db=dbc.get('databasename', "mythconverg"),
    )

def get_filename(chanid, starttime)->str|None:
    with open() as conn:
        with conn.cursor() as c:
            f = _get_filename(c, chanid, starttime)
            if f is not None:
                return f

    log.error(f"No mythtv recording found for {chanid}_{starttime}")
    return None

def _get_filename(cursor, chanid, starttime):
    cursor.execute("SELECT s.dirname, r.basename FROM recorded r, storagegroup s "\
                   "WHERE r.chanid = %s AND r.starttime = %s AND r.storagegroup = s.groupname AND r.hostname = s.hostname",
                   (chanid, starttime))
    for (d,f) in cursor.fetchall():
        if d and f:
            return os.path.join(d,f)
    return None

def get_breaks(chanid, starttime)->list[tuple[float,float]]:
    marks = []
    filename = None

    with _open() as conn:
        with conn.cursor() as c:
            filename = _get_filename(c, chanid, starttime)
            
            c.execute("SELECT mark, type FROM recordedmarkup "\
                        "WHERE r.chanid = %s AND r.starttime = %s AND (type = 4 OR type = 5) "\
                        "ORDER BY mark ASC",
                        (chanid, starttime))
            for (m,t) in c.fetchall():
                marks.append((m,t))

    if not marks or filename is None:
        return []
    
    with av.open(filename) as container:
        container.gen_pts = True
        container.discard_corrupt = True
        try:
            n = 0
            for f in container.decode(video=0):
                n += 1
                if n >= 300:
                    break
        except:
            pass
        #duration = container.duration / av.time_base
        rate = container.streams.video[0].guessed_rate
    
    result = []
    for (m,t) in marks:
        v = float(m/rate)
        if t == 4:
            result.append((v,None))
        else:
            result[-1] = (result[-1][0], v)
    return result
