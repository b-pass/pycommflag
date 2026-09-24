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
        autocommit=True
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

# Returns (frame rate, duration in seconds); rate is None if the file can't be probed.
def _probe_video(filename)->tuple[float|None,float]:
    try:
        with av.open(filename) as container:
            try:
                for f in container.decode(video=0):
                    break
            except Exception:
                pass
            duration = container.duration / av.time_base if container.duration else 0
            rate = container.streams.video[0].average_rate
            return (float(rate) if rate else None, duration)
    except Exception:
        log.exception(f"Could not probe '{filename}'")
        return (None, 0)

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
        
        rate = _probe_video(filename)[0]
        if not rate:
            log.error(f"No frame rate for '{filename}', cannot convert mythtv marks to times")
            return []
        
        c.execute("SELECT mark, type FROM recordedmarkup "\
                  "WHERE chanid = %s AND starttime = %s AND (type = 4 OR type = 5) "\
                  "ORDER BY mark ASC",
                  (chanid, starttime))
        for (m,t) in c.fetchall():
            guess = None
            with conn.cursor() as tc:
                tc.execute("SELECT `offset`,mark FROM recordedseek "\
                           "WHERE chanid = %s and starttime = %s AND type = 33 "\
                           "ORDER BY ABS(CAST(mark AS SIGNED) - "+str(int(m))+") ASC "\
                           "LIMIT 1",
                           (chanid,starttime))
                for (o,om) in tc.fetchall():
                    guess = float(o)/1000 + (int(m) - int(om))/rate
                    break
            if guess is None:
                guess = int(m)/rate
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

# MythTV stores commbreaks as frame numbers in its DB
# Which is from like 1999
# But we're using times instead.
# MythTV stores the time associated with each keyframe in the DB as type 33.
# So we find the frame number of the timestamp closest to the one we want and 
# then use the frame rate to skip to the exact frame number we have flagged.
# Recordings with no type 33 seek table at all just get the frame rate applied.
def _frame_for_time(cursor, chanid, starttime, when, rate)->int:
    cursor.execute("SELECT `offset`,mark FROM recordedseek "\
                   "WHERE chanid = %s AND starttime = %s AND type = 33 "\
                   "ORDER BY ABS(CAST(`offset` AS SIGNED) - "+str(int(when*1000))+") ASC "\
                   "LIMIT 1",
                   (chanid, starttime))
    row = cursor.fetchone()
    if row is None:
        frame = round(when * rate)
    else:
        (o,m) = row
        frame = int(m) + round((when - float(o)/1000) * rate)
    return frame if frame > 0 else 0

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
        duration = 0
        if flog:
            rate = flog.get("frame_rate", None)
            duration = flog.get("duration", 0)
        if not rate:
            (rate, duration) = _probe_video(filename)
        if not rate:
            log.error(f"No frame rate for '{filename}', cannot convert marks to frame numbers")
            return False
        rate = float(rate)
        
        c.execute("DELETE FROM recordedmarkup "\
                  "WHERE chanid = %s AND starttime = %s AND (type = 4 OR type = 5) ",
                  (chanid, starttime))
        
        for (st,(b,e)) in marks:
            if not isinstance(st, SceneType):
                st = SceneType(int(st))
            if st == SceneType.DO_NOT_USE:
                continue
            #print(st,b,e)

            fb = _frame_for_time(c, chanid, starttime, b, rate)
            fe = _frame_for_time(c, chanid, starttime, e, rate)
            
            if fb >= fe: # sanity
                continue
            
            if st == SceneType.COMMERCIAL:
                nbreaks += 1
                log.debug(f".... {st} {fb} {fe}")
                try:
                    c.execute("INSERT INTO recordedmarkup (chanid,starttime,mark,type) "\
                            "VALUES(%s,%s,%s,4),(%s,%s,%s,5);",
                            (chanid, starttime, fb, chanid, starttime, fe))
                except Exception:
                    log.exception("error adding break segment")
        
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
        c.execute('UPDATE jobqueue '
                  'SET comment = %s, status = %s '
                  'WHERE id = %s', (msg, status, opts.mythjob))


def check_method(chanid):
    try:
        conn = _open()
        if conn is not None:
            with conn.cursor() as c:
                c.execute("SELECT commmethod FROM channel "\
                        "WHERE chanid = %s "\
                        "LIMIT 1",
                        (chanid,))
                (m,) = c.fetchone()
                if int(m) == 0:
                    return False
    except:
        pass
    return True
