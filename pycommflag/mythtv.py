import sys
import os
import xml.etree.ElementTree as xml
import MySQLdb as mysql
import logging as log

def get_filename(chanid, starttime):
    dbc = {}
    cfgfile = os.path.join(os.path.expanduser('~'), '.mythtv/config.xml')
    for e in xml.parse(cfgfile).find('Database').iter():
        dbc[e.tag.lower()] = e.text
    conn = mysql.connect(
        host=dbc.get('host', "localhost"),
        user=dbc.get('username', "mythtv"),
        passwd=dbc.get('password', "mythtv"),
        db=dbc.get('databasename', "mythconverg"),
    )

    with conn.cursor() as c:
        c.execute("SELECT s.dirname, r.basename FROM recorded r, storagegroup s "\
                  "WHERE r.chanid = %s AND r.starttime = %s AND r.storagegroup = s.groupname AND r.hostname = s.hostname",
                  (chanid, starttime))
        for (d,f) in c.fetchall():
            if d and f:
                return os.path.join(d,f)
    
    log.error(f"No mythtv recording found for {chanid}_{starttime}")
    return None
