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
        
        rate = 29.97
        with av.open(filename) as container:
            try:
                for f in container.decode(video=0):
                    break
            except:
                pass
            #duration = container.duration / av.time_base
            rate = container.streams.video[0].average_rate
        
        c.execute("SELECT mark, type FROM recordedmarkup "\
                  "WHERE chanid = %s AND starttime = %s AND (type = 4 OR type = 5) "\
                  "ORDER BY mark ASC",
                  (chanid, starttime))
        for (m,t) in c.fetchall():
            guess = 0
            with conn.cursor() as tc:
                tc.execute("SELECT `offset`,mark AS o FROM recordedseek "\
                           "WHERE chanid = %s and starttime = %s AND type = 33 "\
                           "ORDER BY ABS(CAST(mark AS SIGNED) - "+str(int(m))+") ASC "\
                           "LIMIT 1",
                           (chanid,starttime))
                for (o,om) in tc.fetchall():
                    guess = float(o)/1000 + (int(m) - int(om))/rate
                    break
            if not guess and m >= 30:
                guess = m/rate
            marks.append((guess,t))
        
        if not marks:
            return []
    
    result = []
    for (v,t) in marks:
        if t == 4:
            result.append((v,None))
        else:
            result[-1] = (result[-1][0], v)
    return result

def set_breaks(chanid, starttime, marks)->None:
    conn = _open()
    if conn is None:
        return
    
    with conn.cursor() as c:
        filename = _get_filename(c, chanid, starttime)
        if not filename:
            return

        rate = 29.97
        with av.open(filename) as container:
            try:
                for f in container.decode(video=0):
                    break
            except:
                pass
            #duration = container.duration / av.time_base
            rate = container.streams.video[0].average_rate
        
        c.execute("DELETE FROM recordedmarkup "\
                  "WHERE chanid = %s AND starttime = %s AND (type = 4 OR type = 5) ",
                  (chanid, starttime))
        
        for (st,(b,e)) in marks:
            print(st,b,e)
            if st not in [SceneType.COMMERCIAL, SceneType.COMMERCIAL.value]:
                # TODO: other tag types (intro, outro, etc)?? are they chapters? commercials?
                continue
            c.execute("SELECT `offset`,mark FROM recordedseek "\
                      "WHERE chanid = %s AND starttime = %s AND type = 33 "\
                      "ORDER BY ABS(CAST(`offset` AS SIGNED) - "+str(int(e*1000))+") ASC "\
                      "LIMIT 1",
                      (chanid, starttime))
            o = 0
            m = 0
            for (o,m) in c.fetchall():
                break
            fb = int(m) + round((b - float(o)/1000) * rate)
            if fb < 0: fb = 0

            c.execute("SELECT `offset`,mark FROM recordedseek "\
                      "WHERE chanid = %s AND starttime = %s AND type = 33 "\
                      "ORDER BY ABS(CAST(`offset` AS SIGNED) - "+str(int(e*1000))+") ASC "\
                      "LIMIT 1",
                      (chanid, starttime))
            o = 0
            m = 0
            for (o,m) in c.fetchall():
                break
            fe = m + round((e - o/1000) * rate)

            print("\t....",st,fb,fe)
            
            c.execute("INSERT INTO recordedmarkup (chanid,starttime,mark,type) "\
                      "VALUES(%s,%s,%s,4),(%s,%s,%s,5);",
                      (chanid, starttime, fb, chanid, starttime, fe))
        