import sys
import os
import logging as log
import av
from .feature_span import SceneType

g_connection = None
g_off = False
def _open():
    global g_connection
    global g_off
    if g_connection is not None or g_off:
        return g_connection
    
    dbc = {}
    cfgfile = os.path.join(os.path.expanduser('~'), '.mythtv/config.xml')
    if not os.path.exists(cfgfile):
        log.info(f"No mythtv config file at '{cfgfile}', so no mythtv extensions will work")
        g_off = True
        return None
    import xml.etree.ElementTree as xml
    for e in xml.parse(cfgfile).find('Database').iter():
        dbc[e.tag.lower()] = e.text
    import MySQLdb as mysql
    g_connection = mysql.connect(
        host=dbc.get('host', "localhost"),
        user=dbc.get('username', "mythtv"),
        passwd=dbc.get('password', "mythtv"),
        db=dbc.get('databasename', "mythconverg"),
    )
    return g_connection

def _get_filename(cursor, chanid, starttime):
    cursor.execute("SELECT s.dirname, r.basename FROM recorded r, storagegroup s "\
                   "WHERE r.chanid = %s AND r.starttime = %s AND r.storagegroup = s.groupname AND r.hostname = s.hostname",
                   (chanid, starttime))
    for (d,f) in cursor.fetchall():
        if d and f:
            return os.path.join(d,f)
    return None

def get_filename(opts)->str|None:
    if not opts.chanid or not opts.starttime:
        if opts.mythjob:
            conn = _open()
            if conn is not None:
                with conn.cursor() as c:
                    c.execute("SELECT chanid, starttime FROM jobqueue WHERE id = %s", (opts.mythjob,))
                    for (ci,st) in c.fetchall():
                        f = _get_filename(c, ci, st)
                        if f is not None:
                            if not opts.chanid:
                                opts.chanid = ci
                            if not opts.starttime:
                                opts.starttime = st
                            log.debug(f"Resolved job queue to file {f}")
                            return f
                log.error(f"No mythtv recording found for job {opts.mythjob}")
        return None
    
    chanid = opts.chanid
    starttime = opts.starttime

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

def set_breaks(opts, marks, flog=None)->bool:
    chanid = opts.chanid
    starttime = opts.starttime
    if not chanid or not starttime:
        return False
    
    conn = _open()
    if conn is None:
        return False
    
    with conn.cursor() as c:
        filename = _get_filename(c, chanid, starttime)
        if not filename:
            return False

        log.debug(f"Set breaks in myth DB for {chanid}_{starttime}")

        nbreaks = 0
        rate = None
        if flog:
            rate = flog.get("frame_rate", None)
            duration = flog.get("duration", 0)
        if not rate:
            with av.open(filename) as container:
                try:
                    for f in container.decode(video=0):
                        break
                except:
                    pass
                duration = container.duration / av.time_base
                rate = container.streams.video[0].average_rate
        
        c.execute("DELETE FROM recordedmarkup "\
                  "WHERE chanid = %s AND starttime = %s AND (type = 2 OR type = 4 OR type = 5) ",
                  (chanid, starttime))
        
        intro = None
        cred_done = False
        for (st,(b,e)) in marks:
            if type(st) is int:
                st = SceneType(st)
            if st == SceneType.DO_NOT_USE:
                continue
            #print(st,b,e)

            # MythTV stores commbreaks as frame numbers in its DB
            # Which is from like 1999
            # But we're using times instead.
            # MythTV stores the time associated with each keyframe in the DB as type 33.
            # So we find the frame number of the timestamp closest to the one we want and 
            # then use the frame rate to ship to the exact frame number we have flagged.

            c.execute("SELECT `offset`,mark FROM recordedseek "\
                      "WHERE chanid = %s AND starttime = %s AND type = 33 "\
                      "ORDER BY ABS(CAST(`offset` AS SIGNED) - "+str(int(b*1000))+") ASC "\
                      "LIMIT 1",
                      (chanid, starttime))
            (o,m) = c.fetchone()
            fb = int(m) + round((b - float(o)/1000) * rate)
            if fb < 0: fb = 0

            c.execute("SELECT `offset`,mark FROM recordedseek "\
                      "WHERE chanid = %s AND starttime = %s AND type = 33 "\
                      "ORDER BY ABS(CAST(`offset` AS SIGNED) - "+str(int(e*1000))+") ASC "\
                      "LIMIT 1",
                      (chanid, starttime))
            (o,m) = c.fetchone()
            fe = int(m) + round((e - o/1000) * rate)
            
            if st == SceneType.COMMERCIAL:
                nbreaks += 1
                log.debug(f".... {st} {fb} {fe}")
                c.execute("INSERT INTO recordedmarkup (chanid,starttime,mark,type) "\
                          "VALUES(%s,%s,%s,4),(%s,%s,%s,5);",
                          (chanid, starttime, fb, chanid, starttime, fe))
            elif st == SceneType.INTRO:
                intro = (chanid, starttime,fe)
            elif st == SceneType.CREDITS and not cred_done:
                cred_done = True
                log.debug(f".... {st} {fe} (B)")
                c.execute("INSERT INTO recordedmarkup (chanid,starttime,mark,type) "\
                          "VALUES(%s,%s,%s,2)",
                          (chanid, starttime, fe))
            else:
                pass
        
        if intro is not None:
            log.debug(f".... {SceneType.INTRO} {intro[2]} (B)")
            c.execute("INSERT INTO recordedmarkup (chanid,starttime,mark,type) "\
                        "VALUES(%s,%s,%s,2)", intro)
        
        c.execute("UPDATE recorded SET commflagged = %s "\
                  "WHERE chanid = %s AND starttime = %s", (1 if nbreaks else 0, chanid, starttime))
        
    set_job_status(opts, msg=f'Found {nbreaks} commercial break{"s" if nbreaks != 1 else ""}', status='success')
    if opts.exitcode:
        sys.exit(nbreaks) # yes, this is dumb, but its what the jobqueue code looks for when we run as the CommercialFlag command
    return True

def set_job_status(opts, msg='', status='run'):
    if not opts.mythjob:
        return
    
    conn = _open()
    if conn is None:
        return

    if status == 'start':
        status = 3
    elif status == 'run':
        status = 4
    elif status == 'done':
        status = 256
    elif status == 'finish' or status == 'finished' or status == 'success':
        status = 272 # Finished (Successfully completed)
    elif status == 'abort':
        status = 288
    else:#if status == 'error':
        status = 304 # errored
    
    with conn.cursor() as c:
        c.execute('UPDATE jobqueue SET comment = %s, status = %s WHERE id = %s', (msg, status, opts.mythjob))
