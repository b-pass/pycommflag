import sys
import os
import logging as log
import av
from .feature_span import SceneType

def _open():
    dbc = {}
    cfgfile = os.path.join(os.path.expanduser('~'), '.mythtv/config.xml')
    if not os.path.exists(cfgfile):
        log.debug(f"No mythtv config file at '{cfgfile}', so no mythtv extensions will work")
        return None
    import xml.etree.ElementTree as xml
    for e in xml.parse(cfgfile).find('Database').iter():
        dbc[e.tag.lower()] = e.text
    import MySQLdb as mysql
    return mysql.connect(
        host=dbc.get('host', "localhost"),
        user=dbc.get('username', "mythtv"),
        passwd=dbc.get('password', "mythtv"),
        db=dbc.get('databasename', "mythconverg"),
    )

def _get_filename(cursor, chanid, starttime):
    cursor.execute("SELECT s.dirname, r.basename FROM recorded r, storagegroup s "\
                   "WHERE r.chanid = %s AND r.starttime = %s AND r.storagegroup = s.groupname AND r.hostname = s.hostname",
                   (chanid, starttime))
    for (d,f) in cursor.fetchall():
        if d and f:
            return os.path.join(d,f)
    return None

def get_filename(chanid, starttime)->str|None:
    conn = _open()
    if conn is None:
        return None
    
    with conn.cursor() as c:
        f = _get_filename(c, chanid, starttime)
        if f is not None:
            return f

    log.error(f"No mythtv recording found for {chanid}_{starttime}")
    return None

def get_breaks(chanid, starttime)->list[tuple[float,float]]:
    marks = []
    filename = None
    
    conn = _open()
    if conn is None:
        return []
    
    with conn.cursor() as c:
        filename = _get_filename(c, chanid, starttime)
        if not filename:
            return []
        
        c.execute("SELECT mark, type FROM recordedmarkup "\
                  "WHERE chanid = %s AND starttime = %s AND (type = 4 OR type = 5) "\
                  "ORDER BY mark ASC",
                  (chanid, starttime))
        for (m,t) in c.fetchall():
            marks.append((m,t))
        
        if not marks:
            return []
    
    with av.open(filename) as container:
        try:
            n = 0
            for _ in container.decode(video=0):
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

def set_breaks(chanid, starttime, marks)->None:
    filename = None
    
    conn = _open()
    if conn is None:
        return
    
    rate = 29.97
    with conn.cursor() as c:
        filename = _get_filename(c, chanid, starttime)
        if not filename:
            return
        with av.open(filename) as container:
            try:
                n = 0
                for _ in container.decode(video=0):
                    n += 1
                    if n >= 300:
                        break
            except:
                pass
            #duration = container.duration / av.time_base
            rate = container.streams.video[0].guessed_rate
        
        c.execute("DELETE FROM recordedmarkup "\
                  "WHERE chanid = %s AND starttime = %s AND (type = 4 OR type = 5) ",
                  (chanid, starttime))
        
        # TODO: other tag types (intro, outro, etc)?? are they chapters? commercials?
        for (st,(b,e)) in marks:
            if st in [st == SceneType.COMMERCIAL,SceneType.COMMERCIAL.value]:
                fb = round(b*rate)
                fe = round(e*rate)
                c.execute("INSERT INTO recordedmarkup (chanid,starttime,mark,type) "\
                          "VALUES(%s,%s,%s,4), VALUES(%s,%s,%s,5);",
                          (chanid, starttime, fb, chanid, starttime, fe))
        